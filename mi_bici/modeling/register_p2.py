"""Registra el modelo elegido de P2, documenta su linaje y verifica el alias champion.

Uso::

    python -m mi_bici.modeling.register_p2 --run-id <RUN_ID>

El run de origen debe ser una corrida candidata de P2 terminada.
"""

import json

import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer

from mi_bici.config import MLFLOW_TRACKING_URI

MODEL_NAME = "mibici-equipo02-p2-riesgo"
MODEL_ARTIFACT = "model"
ALIAS = "champion"

app = typer.Typer()


def lineage_from_run(run) -> dict[str, str]:
    """Reúne el linaje del modelo a partir de los tags y parámetros del run."""

    tags = run.data.tags
    params = run.data.params

    return {
        "question": "P2",
        "experiment_id": run.info.experiment_id,
        "source_run_id": run.info.run_id,
        "source_run_name": tags.get("mlflow.runName", ""),
        "git_commit": tags.get("git_commit", ""),
        "data_version": tags.get("features_data_version", ""),
        "model_family": tags.get("model_family", ""),
        "ablation": params.get("ablation", ""),
        "C": params.get("C", ""),
        "class_weight": params.get("class_weight", ""),
        "risk_percentile": params.get("risk_percentile", ""),
        "prediction_horizon": params.get("prediction_horizon", ""),
        "unit_of_analysis": tags.get("unit_of_analysis", ""),
    }


def validate_source(run) -> None:
    """Exige que el run sea una corrida candidata de P2 terminada."""

    if run.info.status != "FINISHED":
        raise ValueError("La corrida de origen debe estar terminada.")

    if run.data.tags.get("question") != "P2":
        raise ValueError("La corrida de origen debe corresponder a P2.")

    if run.data.tags.get("stage_of_work") != "candidate":
        raise ValueError("Solo se registran corridas candidatas (stage_of_work=candidate).")


@app.command()
def main(
    run_id: str = typer.Option(
        ...,
        help="Run de la corrida candidata seleccionada.",
    ),
    name: str = typer.Option(
        MODEL_NAME,
        help="Nombre del modelo en el Registry.",
    ),
) -> None:
    """Registra la corrida, documenta su linaje y asigna el alias champion."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    # ---------------------------------------------------------
    # 1. Obtener y validar corrida candidata
    # ---------------------------------------------------------

    origin = client.get_run(run_id)

    validate_source(origin)

    lineage = lineage_from_run(origin)

    source_uri = f"runs:/{run_id}/{MODEL_ARTIFACT}"

    typer.echo("")
    typer.echo("===== P2 MODEL REGISTRATION =====")
    typer.echo(f"Run seleccionado: {run_id}")
    typer.echo(f"Model URI: {source_uri}")

    # ---------------------------------------------------------
    # 2. Verificar que el modelo exista en el run
    # ---------------------------------------------------------

    try:
        original = mlflow.sklearn.load_model(source_uri)
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar el modelo desde {source_uri}.") from exc

    typer.echo("Modelo original cargado correctamente.")

    # ---------------------------------------------------------
    # 3. Buscar si ya existe una versión para este run
    # ---------------------------------------------------------

    existing = [
        version
        for version in client.search_model_versions(f"name = '{name}'")
        if version.run_id == run_id
    ]

    if existing:
        version = max(
            existing,
            key=lambda item: int(item.version),
        ).version

        typer.echo(f"Se reutiliza la versión existente: {version}")

    else:
        typer.echo(f"Registrando modelo en Registry: {name}")

        version = mlflow.register_model(
            source_uri,
            name,
        ).version

        typer.echo(f"Versión registrada: {version}")

    # ---------------------------------------------------------
    # 4. Obtener métricas de la corrida
    # ---------------------------------------------------------

    metrics = {key: round(value, 4) for key, value in sorted(origin.data.metrics.items())}

    # ---------------------------------------------------------
    # 5. Documentar modelo registrado
    # ---------------------------------------------------------

    client.update_registered_model(
        name=name,
        description=(
            "Modelo P2 de MiBici para clasificación de riesgo "
            "operativo a nivel estación-hora. "
            "Predice si una estación presenta un nivel de "
            "desbalance operacional alto, definido mediante "
            "el percentil de riesgo calculado sobre TRAIN. "
            "El modelo seleccionado utiliza regresión logística "
            "con el conjunto de variables correspondiente a la "
            "ablación seleccionada."
        ),
    )

    lineage_text = "\n".join(f"{key}: {value}" for key, value in lineage.items())

    metrics_text = "\n".join(f"{key}: {value}" for key, value in metrics.items())

    client.update_model_version(
        name=name,
        version=version,
        description=(
            "Modelo P2 seleccionado después de comparar "
            "el baseline y las variantes candidatas mediante "
            "métricas de validación temporal. "
            "La selección priorizó Recall debido al costo "
            "operativo de no detectar estaciones en riesgo.\n\n"
            "LINaje:\n"
            f"{lineage_text}\n\n"
            "Métricas registradas:\n"
            f"{metrics_text}"
        ),
    )

    # ---------------------------------------------------------
    # 6. Guardar tags de linaje en Model Registry
    # ---------------------------------------------------------

    for key, value in lineage.items():
        if value != "":
            client.set_model_version_tag(
                name,
                version,
                key,
                str(value),
            )

    # ---------------------------------------------------------
    # 7. Verificar que el modelo registrado pueda cargarse
    # ---------------------------------------------------------

    registered_uri = f"models:/{name}/{version}"

    typer.echo(f"Cargando modelo registrado: {registered_uri}")

    registered = mlflow.sklearn.load_model(registered_uri)

    typer.echo("Modelo registrado cargado correctamente.")

    # ---------------------------------------------------------
    # 8. Verificación básica original vs registrado
    #
    # Utilizamos el input_example almacenado en MLflow.
    # ---------------------------------------------------------

    input_example_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{source_uri}/input_example.json"
    )

    with open(input_example_path, "r", encoding="utf-8") as f:
        input_example_data = json.load(f)

    input_example = pd.DataFrame(
        input_example_data["data"],
        columns=input_example_data["columns"],
    )

    if input_example is not None:
        if isinstance(input_example, (dict, list)):
            example = pd.DataFrame(input_example)

        else:
            example = input_example

        original_prediction = original.predict(example)

        registered_prediction = registered.predict(example)

        np.testing.assert_array_equal(
            original_prediction,
            registered_prediction,
        )

        typer.echo("OK: original y registrado producen las mismas predicciones.")

    else:
        typer.echo(
            "Advertencia: no se encontró input_example "
            "para realizar la comparación de predicciones."
        )

    # ---------------------------------------------------------
    # 9. Asignar alias champion
    # ---------------------------------------------------------

    client.set_registered_model_alias(
        name,
        ALIAS,
        version,
    )

    resolved = client.get_model_version_by_alias(
        name,
        ALIAS,
    )

    if str(resolved.version) != str(version):
        raise RuntimeError("El alias champion no apunta a la versión esperada.")

    typer.echo(f"OK: alias {ALIAS} apunta a versión {version}.")

    # ---------------------------------------------------------
    # 10. Verificar carga mediante alias champion
    # ---------------------------------------------------------

    alias_uri = f"models:/{name}@{ALIAS}"

    champion = mlflow.sklearn.load_model(alias_uri)

    typer.echo(f"OK: modelo champion cargado desde {alias_uri}.")

    if input_example is not None:
        champion_prediction = champion.predict(example)

        np.testing.assert_array_equal(
            original_prediction,
            champion_prediction,
        )

        typer.echo("OK: champion produce las mismas predicciones que el modelo original.")

    typer.echo("")
    typer.echo("==========================================")
    typer.echo("P2 registrado correctamente.")
    typer.echo(f"Versión: {version}")
    typer.echo(f"URI: {alias_uri}")
    typer.echo("==========================================")


if __name__ == "__main__":
    app()
