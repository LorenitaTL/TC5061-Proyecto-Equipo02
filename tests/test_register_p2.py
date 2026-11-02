from types import SimpleNamespace

import pytest

from mi_bici.modeling.register_p2 import lineage_from_run, validate_source


def make_run(status="FINISHED", question="P2", stage_of_work="candidate", **tags):
    """Run falso con los tags y parámetros que registra train_p2 para HistGradientBoosting."""
    run_tags = {
        "mlflow.runName": "P2-hgb-candidate",
        "question": question,
        "stage_of_work": stage_of_work,
        "git_commit": "abc123",
        "data_version": "v1",
        "model_family": "HistGradientBoosting",
        "ablation": "completo",
    }
    run_tags.update(tags)
    return SimpleNamespace(
        info=SimpleNamespace(status=status, run_id="run-1", experiment_id="7"),
        data=SimpleNamespace(
            tags={key: value for key, value in run_tags.items() if value is not None},
            params={
                "features_data_version": "sha-features",
                "train_end": "2026-04-01",
                "validation_end": "2026-06-01",
                "risk_percentile": "0.8",
                "learning_rate": "0.1",
                "max_iter": "200",
            },
        ),
    )


def test_lineage_from_run_collects_origin_commit_and_data_version():
    lineage = lineage_from_run(make_run())

    assert lineage["source_run_id"] == "run-1"
    assert lineage["experiment_id"] == "7"
    assert lineage["git_commit"] == "abc123"
    assert lineage["data_version"] == "v1"
    assert lineage["features_data_version"] == "sha-features"
    assert lineage["train_end"] == "2026-04-01"


def test_lineage_from_run_keeps_only_the_model_hyperparameters():
    """Un modelo HistGradientBoosting no lleva C ni class_weight de la regresión logística."""
    lineage = lineage_from_run(make_run())

    assert lineage["learning_rate"] == "0.1"
    assert "C" not in lineage
    assert "class_weight" not in lineage
    assert "" not in lineage.values()


def test_validate_source_accepts_finished_p2_candidate():
    validate_source(make_run())


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"status": "FAILED"}, "terminada"),
        ({"question": "P1"}, "P2"),
        ({"stage_of_work": "baseline"}, "candidatas"),
    ],
)
def test_validate_source_rejects_other_runs(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_source(make_run(**overrides))


@pytest.mark.parametrize(
    "missing_tags, message",
    [
        ({"git_commit": None}, "git_commit"),
        ({"data_version": ""}, "data_version"),
        ({"git_commit": None, "data_version": None}, "git_commit, data_version"),
    ],
)
def test_validate_source_rejects_incomplete_lineage(missing_tags, message):
    """Un run sin commit o sin versión de datos (o con el tag vacío) no se registra."""
    with pytest.raises(ValueError, match=message):
        validate_source(make_run(**missing_tags))
