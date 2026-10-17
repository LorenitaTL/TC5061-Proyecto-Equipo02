"""Configuración y trazabilidad compartida de experimentos P1."""

import os

import mlflow

from mi_bici.modeling.splits_p1 import TRAIN_END, VALIDATION_END
from mi_bici.modeling.tracking import get_data_version, get_git_commit

DEFAULT_EXPERIMENT_NAME = "mibici-equipo02-P1"
TRACKING_URI = "sqlite:///mlflow.db"


def configure_mlflow() -> str:
    """Configura MLflow y devuelve el nombre del experimento."""
    experiment_name = os.getenv(
        "MIBICI_MLFLOW_EXPERIMENT",
        DEFAULT_EXPERIMENT_NAME,
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    return experiment_name


def log_p1_traceability() -> None:
    """Registra etiquetas y parámetros de trazabilidad de P1."""
    commit = get_git_commit()
    data_version = get_data_version()

    mlflow.set_tags(
        {
            "team": "Equipo02",
            "question": "P1",
            "git_commit": commit,
        }
    )

    mlflow.log_params(
        {
            "data_version": data_version,
            "source_manifest": "references/data_manifest.json",
            "data_start_month": "2025-01",
            "data_end_month": "2026-06",
            "train_start": "2025-01-01",
            "train_end_exclusive": TRAIN_END.strftime("%Y-%m-%d"),
            "validation_start": TRAIN_END.strftime("%Y-%m-%d"),
            "validation_end_exclusive": VALIDATION_END.strftime("%Y-%m-%d"),
            "test_start": VALIDATION_END.strftime("%Y-%m-%d"),
        }
    )
