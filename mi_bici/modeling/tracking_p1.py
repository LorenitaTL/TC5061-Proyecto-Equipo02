"""Configuración y trazabilidad compartida de experimentos P1."""

import hashlib
import json
import os
import subprocess

import mlflow

from mi_bici.config import PROJ_ROOT, REFERENCES_DIR
from mi_bici.modeling.splits_p1 import TRAIN_END, VALIDATION_END

DEFAULT_EXPERIMENT_NAME = "MiBici_P1_Demanda_Horaria"
TRACKING_URI = "sqlite:///mlflow.db"


def get_git_commit() -> str:
    """Obtiene el commit actual del repositorio."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_data_version() -> str:
    """Calcula una huella estable del manifiesto de datos fuente."""
    path = REFERENCES_DIR / "data_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    # La fecha de generación no modifica la identidad de los datos.
    manifest.pop("generated_at_utc", None)

    canonical = json.dumps(
        manifest,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
