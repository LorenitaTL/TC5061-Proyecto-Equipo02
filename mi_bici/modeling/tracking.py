"""Configuración y trazabilidad de MLflow compartida por P1, P2 y P3."""

import hashlib
import json
import subprocess

import mlflow

from mi_bici.config import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
    REFERENCES_DIR,
    TEAM,
)

DATA_MANIFEST = REFERENCES_DIR / "data_manifest.json"


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
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))

    # La fecha de generación no modifica la identidad de los datos.
    manifest.pop("generated_at_utc", None)

    canonical = json.dumps(
        manifest,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def configure_mlflow(experiment_name: str = MLFLOW_EXPERIMENT_NAME) -> str:
    """Apunta MLflow a la base del proyecto y activa el experimento.

    Si el experimento no existe, se crea con sus artefactos en ``mlartifacts/``, que está
    excluido de Git.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri(),
        )

    mlflow.set_experiment(experiment_name)

    return experiment_name


def log_traceability(question: str) -> None:
    """Registra en el run activo el equipo, la pregunta (P1, P2 o P3), el commit y la versión
    de los datos."""
    mlflow.set_tags(
        {
            "team": TEAM,
            "question": question,
            "git_commit": get_git_commit(),
        }
    )
    mlflow.log_params(
        {
            "data_version": get_data_version(),
            "source_manifest": DATA_MANIFEST.relative_to(PROJ_ROOT).as_posix(),
        }
    )
