"""Pruebas del baseline de persistencia de P2."""

import pandas as pd
import pytest

from mi_bici.modeling.baseline_p2 import persistence_metrics, persistence_prediction

THRESHOLD = 0.5


def make_df(current: list[float], target: list[float]) -> pd.DataFrame:
    """Crea filas estación-hora con el desbalance actual y el de la hora siguiente."""
    return pd.DataFrame({"imbalance_ratio": current, "target_imbalance_ratio": target})


def test_prediction_includes_threshold() -> None:
    """Un desbalance igual al umbral se predice como riesgo."""
    df = make_df([0.4, 0.5, 0.6], [0.0, 0.0, 0.0])
    assert persistence_prediction(df, THRESHOLD).tolist() == [0, 1, 1]


def test_prediction_is_binary_and_keeps_index() -> None:
    """La predicción es 0/1 entera y conserva el índice para compararla con la etiqueta."""
    df = make_df([0.9, 0.1], [0.0, 0.0]).set_axis([10, 20])
    prediction = persistence_prediction(df, THRESHOLD)
    assert prediction.dtype.kind == "i"
    assert prediction.index.tolist() == [10, 20]


def test_prediction_uses_current_hour_only() -> None:
    """La regla mira la hora actual; la hora siguiente no influye."""
    df = make_df([0.9, 0.1], [0.0, 0.9])
    assert persistence_prediction(df, THRESHOLD).tolist() == [1, 0]


def test_metrics_hand_computed() -> None:
    """Métricas de un caso con TP=1, FP=2, FN=1 y TN=3."""
    df = make_df(
        current=[0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.05],
        target=[0.9, 0.1, 0.1, 0.8, 0.0, 0.0, 0.0],
    )
    metrics = persistence_metrics(df, THRESHOLD)

    assert metrics["accuracy"] == pytest.approx(4 / 7)
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["f1"] == pytest.approx(0.4)
    # 8 de los 10 pares (positivo, negativo) quedan bien ordenados por imbalance_ratio.
    assert metrics["roc_auc"] == pytest.approx(0.8)


def test_metrics_perfect_persistence() -> None:
    """Si la hora siguiente repite la actual, la regla acierta todo."""
    values = [0.9, 0.7, 0.3, 0.1]
    metrics = persistence_metrics(make_df(values, values), THRESHOLD)
    assert metrics == pytest.approx(
        {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0}
    )


def test_metrics_without_positive_predictions() -> None:
    """Sin predicciones de riesgo, precision vale 0 en lugar de fallar."""
    df = make_df([0.1, 0.2, 0.3], [0.9, 0.0, 0.0])
    metrics = persistence_metrics(df, THRESHOLD)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_metrics_keys_and_types() -> None:
    """Devuelve las cinco métricas que main() registra en MLflow, como float."""
    metrics = persistence_metrics(make_df([0.9, 0.1], [0.9, 0.1]), THRESHOLD)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(isinstance(value, float) for value in metrics.values())
