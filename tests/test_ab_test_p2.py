"""Pruebas del comparador A/B de P2 (``mi_bici.monitoring.ab_test_p2``).

No usan MLflow ni ``data/``: las versiones del Registry se simulan con un cliente falso y las
predicciones con datos sintéticos de semilla fija.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mi_bici.modeling.train_p2 import ablation_features
from mi_bici.monitoring.ab_test_p2 import (
    METRICS,
    bootstrap_deltas,
    build_table,
    compute_metrics,
    decide,
    load_evaluation,
    logged_test_metrics,
    max_logged_gap,
    mcnemar_test,
    predict,
    resolve_model,
    table_to_markdown,
)

THRESHOLD = 0.35


@pytest.fixture
def evaluation():
    """2,000 filas de 40 estaciones con ~20 % de Risk y probabilidades informativas."""
    rng = np.random.default_rng(7)
    n = 2000
    groups = np.repeat(np.arange(40), n // 40)
    y = (rng.random(n) < 0.2).astype(int)
    probabilities = np.clip(0.25 + 0.3 * y + rng.normal(0, 0.15, n), 0, 1)
    return groups, y, probabilities


# ---------------------------------------------------------
# Métricas
# ---------------------------------------------------------


def test_predict_uses_greater_or_equal_threshold():
    probabilities = np.array([0.34, 0.35, 0.36])

    assert predict(probabilities, THRESHOLD).tolist() == [0, 1, 1]


def test_compute_metrics_matches_hand_counts():
    # TP = 2, FN = 1, FP = 1, TN = 2.
    y = np.array([1, 1, 1, 0, 0, 0])
    probabilities = np.array([0.9, 0.6, 0.1, 0.8, 0.2, 0.1])

    metrics = compute_metrics(y, probabilities, THRESHOLD)

    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert set(metrics) == set(METRICS)


# ---------------------------------------------------------
# A contra A: el comparador no debe encontrar diferencias
# ---------------------------------------------------------


def test_a_against_a_has_no_discordant_rows(evaluation):
    _, y, probabilities = evaluation
    predictions = predict(probabilities, THRESHOLD)

    result = mcnemar_test(y, predictions, predictions)

    assert result == {"only_a_correct": 0, "only_b_correct": 0, "p_value": 1.0}


def test_a_against_a_has_zero_width_intervals_and_keeps_a(evaluation):
    groups, y, probabilities = evaluation
    predictions = predict(probabilities, THRESHOLD)

    intervals = bootstrap_deltas(groups, y, predictions, predictions, n_bootstrap=200)

    assert intervals == {"recall": (0.0, 0.0), "precision": (0.0, 0.0), "f1": (0.0, 0.0)}
    assert decide(intervals)[0] == "A"


def test_a_against_a_matches_its_own_logged_metrics(evaluation):
    _, y, probabilities = evaluation
    metrics = compute_metrics(y, probabilities, THRESHOLD)
    intervals = {name: (0.0, 0.0) for name in ("recall", "precision", "f1")}

    table = build_table(metrics, metrics, intervals, metrics, metrics)

    assert max_logged_gap(table) == pytest.approx(0.0)
    assert (table["delta_B_menos_A"] == 0).all()


# ---------------------------------------------------------
# B distinta de A
# ---------------------------------------------------------


def test_mcnemar_detects_b_fixing_a_errors(evaluation):
    _, y, probabilities = evaluation
    predictions_a = predict(probabilities, THRESHOLD)
    # B acierta todas las filas.
    predictions_b = y.copy()

    result = mcnemar_test(y, predictions_a, predictions_b)

    assert result["only_a_correct"] == 0
    assert result["only_b_correct"] == int(np.sum(predictions_a != y))
    assert result["p_value"] < 0.001


def test_bootstrap_interval_is_positive_when_b_finds_more_risk(evaluation):
    groups, y, probabilities = evaluation
    predictions_a = predict(probabilities, THRESHOLD)
    # B detecta además la mitad de los positivos que A no ve.
    predictions_b = predictions_a.copy()
    missed = np.flatnonzero((y == 1) & (predictions_a == 0))
    predictions_b[missed[::2]] = 1

    intervals = bootstrap_deltas(groups, y, predictions_a, predictions_b, n_bootstrap=500)

    assert intervals["recall"][0] > 0


def test_bootstrap_is_reproducible_with_the_same_seed(evaluation):
    groups, y, probabilities = evaluation
    predictions_a = predict(probabilities, THRESHOLD)
    predictions_b = predict(probabilities, 0.5)

    first = bootstrap_deltas(groups, y, predictions_a, predictions_b, n_bootstrap=200)
    second = bootstrap_deltas(groups, y, predictions_a, predictions_b, n_bootstrap=200)

    assert first == second


# ---------------------------------------------------------
# Regla de decisión
# ---------------------------------------------------------


@pytest.mark.parametrize(
    ("recall", "f1", "winner"),
    [
        ((0.01, 0.03), (-0.01, 0.02), "B"),  # Recall mejora y F1 no empeora
        ((0.01, 0.03), (-0.03, -0.01), "A"),  # Recall mejora pero F1 empeora
        ((-0.03, -0.01), (-0.01, 0.01), "A"),  # Recall empeora
        ((-0.01, 0.02), (0.00, 0.02), "A"),  # Sin diferencia significativa
    ],
)
def test_decide_only_promotes_b_with_significant_recall_gain(recall, f1, winner):
    intervals = {"recall": recall, "precision": (0.0, 0.0), "f1": f1}

    assert decide(intervals)[0] == winner


# ---------------------------------------------------------
# Registry
# ---------------------------------------------------------


class FakeClient:
    """Cliente de MLflow con una versión por alias y métricas de test registradas."""

    def __init__(self):
        self.calls = []

    def get_model_version_by_alias(self, name, alias):
        self.calls.append(("alias", name, alias))
        return SimpleNamespace(version=5, run_id="run-a")

    def get_model_version(self, name, version):
        self.calls.append(("version", name, version))
        return SimpleNamespace(version=int(version), run_id="run-b")

    def get_run(self, run_id):
        return SimpleNamespace(
            data=SimpleNamespace(
                metrics={"test_recall": 0.7, "test_f1": 0.5, "validation_recall": 0.9}
            )
        )


def test_resolve_model_uses_alias_or_version_number():
    client = FakeClient()

    by_alias = resolve_model(client, "modelo", "champion", "A")
    by_version = resolve_model(client, "modelo", "6", "B")

    assert (by_alias.version, by_alias.run_id) == ("5", "run-a")
    assert (by_version.version, by_version.run_id) == ("6", "run-b")
    assert client.calls == [("alias", "modelo", "champion"), ("version", "modelo", "6")]


def test_logged_test_metrics_keeps_only_test_metrics():
    assert logged_test_metrics(FakeClient(), "run-a") == {"recall": 0.7, "f1": 0.5}


# ---------------------------------------------------------
# Reporte y datos
# ---------------------------------------------------------


def test_markdown_table_has_one_row_per_metric(evaluation):
    _, y, probabilities = evaluation
    metrics = compute_metrics(y, probabilities, THRESHOLD)
    table = build_table(metrics, metrics, {}, {}, {})

    markdown = table_to_markdown(table)

    lines = markdown.strip().splitlines()
    assert len(lines) == 2 + len(METRICS)
    assert lines[2].startswith("| recall |")
    assert "—" in lines[2]  # sin métricas registradas ni intervalo


def test_load_evaluation_rejects_missing_columns(tmp_path):
    path = tmp_path / "evaluation.parquet"
    pd.DataFrame({"station_id": [1], "Risk": [0]}).to_parquet(path)

    with pytest.raises(ValueError, match="Faltan columnas"):
        load_evaluation(path)


def test_load_evaluation_returns_features_target_and_station(tmp_path):
    features = ablation_features("completo")
    frame = pd.DataFrame({name: [0.0, 1.0] for name in features})
    frame["station_id"] = [10, 11]
    frame["hour"] = pd.Timestamp("2026-06-01")
    frame["Risk"] = [0, 1]
    path = tmp_path / "evaluation.parquet"
    frame.to_parquet(path)

    X, y, groups = load_evaluation(path)

    assert list(X.columns) == features
    assert y.tolist() == [0, 1]
    assert groups.tolist() == [10, 11]
