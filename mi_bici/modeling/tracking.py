"""Configuración y trazabilidad de MLflow compartida por P1, P2 y P3."""

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
import subprocess
from typing import Any

import mlflow
from mlflow import ActiveRun, MlflowClient

from mi_bici.config import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
    REFERENCES_DIR,
    TEAM,
)

DATA_MANIFEST = REFERENCES_DIR / "data_manifest.json"

# Etapa de trabajo de cada corrida, para filtrar en la interfaz de MLflow.
STAGES_OF_WORK = ("baseline", "search", "candidate")


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

    # Sin esta etiqueta, la interfaz de MLflow 3 puede tratar el experimento como GenAI y
    # ocultar la pestaña Runs.
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")

    return experiment_name


def log_traceability(
    question: str,
    model_family: str | None = None,
    stage_of_work: str | None = None,
) -> None:
    """Registra en el run activo el equipo, la pregunta (P1, P2 o P3), el commit y la versión
    de los datos.

    ``model_family`` y ``stage_of_work`` son opcionales para no romper los scripts que todavía
    no migran al contrato de la Entrega 2. ``stage_of_work`` debe ser uno de ``STAGES_OF_WORK``.
    """
    if stage_of_work is not None and stage_of_work not in STAGES_OF_WORK:
        raise ValueError(
            f"stage_of_work desconocido: {stage_of_work}. Usa uno de {STAGES_OF_WORK}."
        )

    data_version = get_data_version()
    tags = {
        "team": TEAM,
        "question": question,
        "git_commit": get_git_commit(),
        # También como tag, porque la interfaz filtra por tags y no solo por parámetros.
        "data_version": data_version,
    }
    if model_family is not None:
        tags["model_family"] = model_family
    if stage_of_work is not None:
        tags["stage_of_work"] = stage_of_work

    mlflow.set_tags(tags)
    mlflow.log_params(
        {
            "data_version": data_version,
            "source_manifest": DATA_MANIFEST.relative_to(PROJ_ROOT).as_posix(),
        }
    )


def log_run_config(config: dict[str, Any], filename: str = "run_config.json") -> None:
    """Guarda en ``config/`` la configuración efectiva con la que se ejecutó la corrida.

    Los valores que no son JSON (rutas, fechas) se guardan como texto.
    """
    mlflow.log_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        f"config/{filename}",
    )


class Search:
    """Búsqueda de hiperparámetros organizada como corrida padre con una hija por combinación."""

    def __init__(self, parent: ActiveRun, question: str, model_family: str) -> None:
        self.parent = parent
        self.question = question
        self.model_family = model_family
        self._param_names: set[str] = set()

    @contextmanager
    def trial(self, params: dict[str, Any], run_name: str | None = None) -> Iterator[ActiveRun]:
        """Abre la corrida hija de una combinación y registra sus hiperparámetros."""
        with mlflow.start_run(run_name=run_name, nested=True) as child:
            log_traceability(self.question, self.model_family, "search")
            mlflow.log_params(params)
            self._param_names.update(params)
            yield child

    def log_best(self, metric: str, greater_is_better: bool = True) -> str:
        """Registra en la corrida padre la mejor combinación según ``metric``.

        Devuelve el ``run_id`` de la mejor corrida hija.
        """
        parent_id = self.parent.info.run_id
        order = "DESC" if greater_is_better else "ASC"
        children = mlflow.search_runs(
            experiment_ids=[self.parent.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
            order_by=[f"metrics.{metric} {order}"],
        )
        if children.empty:
            raise ValueError("La búsqueda no tiene corridas hijas.")

        best = children.iloc[0]
        client = MlflowClient()
        client.set_tag(parent_id, "best_child_run_id", best["run_id"])
        client.log_metric(parent_id, f"best_{metric}", float(best[f"metrics.{metric}"]))
        client.log_metric(parent_id, "n_trials", len(children))
        for name in sorted(self._param_names):
            client.log_param(parent_id, f"best_{name}", best[f"params.{name}"])

        return best["run_id"]


@contextmanager
def nested_search(
    question: str, model_family: str, run_name: str | None = None
) -> Iterator[Search]:
    """Abre la corrida padre de una búsqueda de hiperparámetros.

    Bajo ``mlflow run``, la corrida padre es la del proyecto. Uso::

        with nested_search("P2", "logreg", run_name="p2-logreg-search-C") as search:
            for c in (0.1, 1.0, 10.0):
                with search.trial({"C": c}, run_name=f"p2-logreg-C{c}"):
                    mlflow.log_metric("val_recall", entrenar_y_evaluar(c))
            search.log_best("val_recall")
    """
    with mlflow.start_run(run_name=run_name) as parent:
        log_traceability(question, model_family, "search")
        yield Search(parent, question, model_family)
