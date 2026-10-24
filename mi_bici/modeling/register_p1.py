"""Registra el ganador de P1, documenta su linaje y verifica el alias champion."""

import argparse

import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd

from mi_bici.config import MLFLOW_TRACKING_URI
from mi_bici.features_p1 import DATASET_PATH, FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--name", default="mibici-equipo02-p1-demanda")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    origin = client.get_run(args.run_id)
    evaluation = client.get_run(args.evaluation_run_id)
    source_uri = f"runs:/{args.run_id}/modelo_p1"

    if origin.info.status != "FINISHED" or evaluation.info.status != "FINISHED":
        raise ValueError("Las corridas de entrenamiento y evaluación deben estar finalizadas.")

    if origin.data.tags.get("question") != "P1":
        raise ValueError("La corrida de origen debe corresponder a P1.")

    if evaluation.data.params.get("model_uri") != source_uri:
        raise ValueError("La evaluación no corresponde al modelo indicado.")

    lineage = {
        "question": "P1",
        "experiment_id": origin.info.experiment_id,
        "source_run_id": args.run_id,
        "git_commit": origin.data.tags["git_commit"],
        "data_version": origin.data.tags["data_version"],
        "evaluation_run_id": args.evaluation_run_id,
    }

    # Reutiliza una versión del mismo entrenamiento si ya está registrada.
    existing = [
        version
        for version in client.search_model_versions(f"name = '{args.name}'")
        if version.run_id == args.run_id
    ]

    if existing:
        version = max(existing, key=lambda item: int(item.version)).version
        print("Se reutiliza la versión existente:", version)
    else:
        version = mlflow.register_model(source_uri, args.name).version

    description = (
        "Modelo P1 seleccionado mediante validacion y evaluado en prueba "
        "sin reentrenamiento.\n"
        + "\n".join(f"{key}: {value}" for key, value in lineage.items())
        + "\nMetricas de validacion: "
        + str(origin.data.metrics)
        + "\nMetricas de prueba: "
        + str(evaluation.data.metrics)
    )

    client.update_registered_model(
        name=args.name,
        description=(
            "Prediccion de viajes iniciados por estacion-hora en MiBici. "
            "Representa demanda observada, no demanda insatisfecha."
        ),
    )
    client.update_model_version(
        name=args.name,
        version=version,
        description=description,
    )

    for key, value in lineage.items():
        client.set_model_version_tag(args.name, version, key, str(value))

    # Comprueba la versión antes de asignarle el alias.
    original = mlflow.sklearn.load_model(source_uri)
    registered = mlflow.sklearn.load_model(f"models:/{args.name}/{version}")
    example = pd.read_parquet(DATASET_PATH, columns=FEATURES).head(10).astype("float32")

    np.testing.assert_allclose(
        registered.predict(example),
        original.predict(example),
    )

    client.set_registered_model_alias(args.name, "champion", version)
    resolved = client.get_model_version_by_alias(args.name, "champion")
    if str(resolved.version) != str(version):
        raise RuntimeError("El alias no apunta a la versión esperada.")

    alias_uri = f"models:/{args.name}@champion"
    champion = mlflow.sklearn.load_model(alias_uri)
    np.testing.assert_allclose(
        champion.predict(example),
        original.predict(example),
    )

    print("OK: linaje, carga y alias champion verificados.")
    print("Version:", version)
    print("URI:", alias_uri)


if __name__ == "__main__":
    main()
