"""Registro de manifiesto y datos de entrenamiento en una corrida activa."""

import mlflow
import pandas as pd

from mi_bici.modeling.tracking import DATA_MANIFEST, get_data_version


def log_data_manifest() -> str:
    """Guarda el manifiesto y registra su versión en la corrida activa."""
    if mlflow.active_run() is None:
        raise RuntimeError("Se requiere una corrida activa de MLflow.")

    data_version = get_data_version()
    mlflow.log_artifact(str(DATA_MANIFEST), artifact_path="data")
    mlflow.set_tag("data_version", data_version)
    mlflow.log_param("data_version", data_version)

    return data_version


def log_training_data(
    train_data: pd.DataFrame,
    *,
    name: str,
    source: str,
    target: str | None = None,
) -> str:
    """Registra el conjunto usado para entrenar y su manifiesto.

    El llamador debe proporcionar únicamente la partición de entrenamiento,
    con las variables del modelo y sin datos personales. Si se indica target,
    esa columna debe estar incluida en train_data.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("Se requiere una corrida activa de MLflow.")

    if train_data.empty:
        raise ValueError("El conjunto de entrenamiento está vacío.")

    if target is not None and target not in train_data.columns:
        raise ValueError(f"La columna objetivo no existe: {target}")

    dataset = mlflow.data.from_pandas(
        train_data,
        source=source,
        name=name,
        targets=target,
    )
    data_version = log_data_manifest()
    mlflow.log_input(dataset, context="training")

    return data_version
