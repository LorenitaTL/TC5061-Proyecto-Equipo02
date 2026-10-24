from types import SimpleNamespace

import pytest

from mi_bici.modeling.register_p3 import lineage_from_run, validate_source


def make_run(status="FINISHED", question="P3", stage_of_work="candidate"):
    """Run falso con los tags y parámetros que registra train_p3."""
    return SimpleNamespace(
        info=SimpleNamespace(status=status, run_id="run-1", experiment_id="7"),
        data=SimpleNamespace(
            tags={
                "mlflow.runName": "p3-kmeans-perfil-k3",
                "question": question,
                "stage_of_work": stage_of_work,
                "git_commit": "abc123",
                "data_version": "v1",
                "model_family": "kmeans",
            },
            params={
                "n_clusters": "3",
                "features": "balance_am,balance_pm,share_fin_semana",
                "train_start_month": "2025-01",
                "train_end_month": "2025-12",
                "test_start_month": "2026-01",
                "test_end_month": "2026-06",
            },
        ),
    )


def test_lineage_from_run_collects_origin_commit_and_data_version():
    lineage = lineage_from_run(make_run())

    assert lineage["source_run_id"] == "run-1"
    assert lineage["experiment_id"] == "7"
    assert lineage["git_commit"] == "abc123"
    assert lineage["data_version"] == "v1"
    assert lineage["train_period"] == "2025-01 a 2025-12"


def test_validate_source_accepts_finished_p3_candidate():
    validate_source(make_run())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"status": "FAILED"}, "terminada"),
        ({"question": "P1"}, "P3"),
        ({"stage_of_work": "baseline"}, "candidatas"),
    ],
)
def test_validate_source_rejects_other_runs(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_source(make_run(**overrides))
