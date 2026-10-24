"""Registra el modelo elegido de P3, documenta su linaje y verifica el alias champion.

Uso::

    python -m mi_bici.modeling.register_p3 --run-id <RUN_ID>

El run de origen debe ser una corrida candidata de P3 terminada. La comprobación de carga usa
las asignaciones de prueba guardadas en el propio run (``data/station_clusters_test.csv``),
por lo que no requiere los datos procesados.
"""

import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer

from mi_bici.config import MLFLOW_TRACKING_URI

MODEL_NAME = "mibici-equipo02-p3-grupos"
MODEL_ARTIFACT = "modelo_p3"
ALIAS = "champion"

app = typer.Typer()


def lineage_from_run(run) -> dict[str, str]:
    """Reúne el linaje del modelo a partir de los tags y parámetros del run de origen."""
    tags, params = run.data.tags, run.data.params
    return {
        "question": "P3",
        "experiment_id": run.info.experiment_id,
        "source_run_id": run.info.run_id,
        "source_run_name": tags.get("mlflow.runName", ""),
        "git_commit": tags["git_commit"],
        "data_version": tags["data_version"],
        "model_family": tags["model_family"],
        "n_clusters": params["n_clusters"],
        "features": params["features"],
        "train_period": f"{params['train_start_month']} a {params['train_end_month']}",
        "test_period": f"{params['test_start_month']} a {params['test_end_month']}",
    }


def validate_source(run) -> None:
    """Exige que el run sea una corrida candidata de P3 terminada."""
    if run.info.status != "FINISHED":
        raise ValueError("La corrida de origen debe estar terminada.")
    if run.data.tags.get("question") != "P3":
        raise ValueError("La corrida de origen debe corresponder a P3.")
    if run.data.tags.get("stage_of_work") != "candidate":
        raise ValueError("Solo se registran corridas candidatas (stage_of_work=candidate).")


@app.command()
def main(
    run_id: str = typer.Option(..., help="Run de la corrida candidata seleccionada."),
    name: str = typer.Option(MODEL_NAME, help="Nombre del modelo en el Registry."),
) -> None:
    """Registra la corrida, documenta su linaje y asigna el alias champion."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    origin = client.get_run(run_id)
    validate_source(origin)
    lineage = lineage_from_run(origin)
    source_uri = f"runs:/{run_id}/{MODEL_ARTIFACT}"

    # Reutiliza una versión del mismo entrenamiento si ya está registrada.
    existing = [
        version
        for version in client.search_model_versions(f"name = '{name}'")
        if version.run_id == run_id
    ]
    if existing:
        version = max(existing, key=lambda item: int(item.version)).version
        typer.echo(f"Se reutiliza la versión existente: {version}")
    else:
        version = mlflow.register_model(source_uri, name).version

    metrics = {key: round(value, 4) for key, value in sorted(origin.data.metrics.items())}
    client.update_registered_model(
        name=name,
        description=(
            "Grupos de estaciones de MiBici por perfil de uso: balance de flujo en horas pico "
            "y uso en fin de semana. Pipeline StandardScaler + KMeans; devuelve el número de "
            "grupo de cada estación. Uso descriptivo para planeación; no predice demanda."
        ),
    )
    client.update_model_version(
        name=name,
        version=version,
        description=(
            "Modelo P3 seleccionado frente al baseline de ubicación y a las variantes con más "
            "grupos o variables, por su silhouette, su estabilidad entre semillas y la "
            "permanencia de los grupos en el periodo de prueba.\n"
            + "\n".join(f"{key}: {value}" for key, value in lineage.items())
            + f"\nMétricas: {metrics}"
        ),
    )
    for key, value in lineage.items():
        client.set_model_version_tag(name, version, key, str(value))

    # Comprueba la versión antes de asignarle el alias.
    features = lineage["features"].split(",")
    assignments = pd.read_csv(
        mlflow.artifacts.download_artifacts(f"runs:/{run_id}/data/station_clusters_test.csv"),
        index_col=0,
    )
    example = assignments[features]
    expected = assignments["cluster"].to_numpy()

    original = mlflow.sklearn.load_model(source_uri)
    registered = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    np.testing.assert_array_equal(original.predict(example), expected)
    np.testing.assert_array_equal(registered.predict(example), expected)

    client.set_registered_model_alias(name, ALIAS, version)
    resolved = client.get_model_version_by_alias(name, ALIAS)
    if str(resolved.version) != str(version):
        raise RuntimeError("El alias no apunta a la versión esperada.")

    alias_uri = f"models:/{name}@{ALIAS}"
    champion = mlflow.sklearn.load_model(alias_uri)
    np.testing.assert_array_equal(champion.predict(example), expected)

    typer.echo("OK: linaje, carga y alias champion verificados.")
    typer.echo(f"Versión: {version}")
    typer.echo(f"URI: {alias_uri}")


if __name__ == "__main__":
    app()
