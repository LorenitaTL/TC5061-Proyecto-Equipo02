import json

import mlflow
from mlflow import MlflowClient
import pytest

from mi_bici.modeling.tracking import log_run_config, log_traceability, nested_search


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    """Experimento aislado en una base SQLite temporal, sin tocar ``mlflow.db``."""
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment(
        "test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    yield experiment_id
    mlflow.end_run()
    mlflow.set_tracking_uri(previous_uri)


def test_log_traceability_sets_contract_tags(experiment):
    with mlflow.start_run() as run:
        log_traceability("P3", "kmeans", "candidate")

    data = MlflowClient().get_run(run.info.run_id).data
    assert data.tags["team"] == "equipo02"
    assert data.tags["question"] == "P3"
    assert data.tags["model_family"] == "kmeans"
    assert data.tags["stage_of_work"] == "candidate"
    assert data.tags["data_version"] == data.params["data_version"]
    assert len(data.tags["git_commit"]) == 40


def test_log_traceability_keeps_legacy_call(experiment):
    with mlflow.start_run() as run:
        log_traceability("P1")

    tags = MlflowClient().get_run(run.info.run_id).data.tags
    assert tags["question"] == "P1"
    assert "model_family" not in tags
    assert "stage_of_work" not in tags


def test_log_traceability_rejects_unknown_stage(experiment):
    with mlflow.start_run(), pytest.raises(ValueError, match="stage_of_work desconocido"):
        log_traceability("P3", "kmeans", "produccion")


def test_log_run_config_saves_json_under_config(experiment, tmp_path):
    with mlflow.start_run() as run:
        log_run_config({"n_clusters": 4, "input_path": tmp_path})

    path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="config/run_config.json", dst_path=tmp_path
    )
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == {"n_clusters": 4, "input_path": str(tmp_path)}


def test_nested_search_logs_children_and_best_trial(experiment):
    with nested_search("P2", "logreg", run_name="p2-search") as search:
        for c, recall in [(0.1, 0.6), (1.0, 0.8), (10.0, 0.7)]:
            with search.trial({"C": c}, run_name=f"p2-C{c}"):
                mlflow.log_metric("val_recall", recall)
        best_id = search.log_best("val_recall")

    client = MlflowClient()
    parent = client.get_run(search.parent.info.run_id).data
    children = client.search_runs(
        [experiment], filter_string=f"tags.mlflow.parentRunId = '{search.parent.info.run_id}'"
    )

    assert len(children) == 3
    assert all(child.data.tags["stage_of_work"] == "search" for child in children)
    assert parent.tags["stage_of_work"] == "search"
    assert parent.tags["best_child_run_id"] == best_id
    assert parent.metrics == {"best_val_recall": 0.8, "n_trials": 3}
    assert parent.params["best_C"] == "1.0"


def test_nested_search_resumes_mlflow_project_run(experiment, monkeypatch):
    """Bajo ``mlflow run``, la corrida padre debe ser la que creó el proyecto."""
    project_run = MlflowClient().create_run(experiment)
    monkeypatch.setenv("MLFLOW_RUN_ID", project_run.info.run_id)

    with (
        nested_search("P2", "logreg", run_name="p2-search") as search,
        search.trial({"C": 1.0}) as child,
    ):
        pass

    assert search.parent.info.run_id == project_run.info.run_id
    child_tags = MlflowClient().get_run(child.info.run_id).data.tags
    assert child_tags["mlflow.parentRunId"] == project_run.info.run_id


def test_log_best_fails_without_children(experiment):
    with nested_search("P2", "logreg") as search, pytest.raises(ValueError, match="hijas"):
        search.log_best("val_recall")
