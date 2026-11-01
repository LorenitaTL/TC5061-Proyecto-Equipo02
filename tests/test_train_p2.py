"""Pruebas del entrenamiento y preparación del modelo P2."""

import numpy as np
import pandas as pd
import pytest

from mi_bici.modeling.train_p2 import (
    ABLATIONS,
    FEATURE_COLUMNS,
    ablation_features,
    ablation_run_name,
    create_risk_label,
    evaluate_model,
    temporal_split,
)


def test_ablation_feature_counts() -> None:
    """Cada ablación conserva el número de variables reportado en el README."""
    assert len(ablation_features("completo")) == 30
    assert len(ablation_features("sin_desbalance_actual")) == 28
    assert len(ablation_features("sin_flujo_actual")) == 25


def test_ablations_only_remove_existing_features() -> None:
    """Las variables quitadas existen en el modelo completo."""
    for removed in ABLATIONS.values():
        assert set(removed) <= set(FEATURE_COLUMNS)


def test_ablation_run_names() -> None:
    """El nombre del run refleja la familia de modelo y la ablación."""
    assert (
        ablation_run_name("completo", model_family="HistGradientBoosting")
        == "P2-HistGradientBoosting-candidate"
    )
    assert (
        ablation_run_name("sin_temporal", model_family="HistGradientBoosting")
        == "P2-HistGradientBoosting-sin-temporal"
    )
    assert (
        ablation_run_name("completo", model_family="LogisticRegression")
        == "P2-LogisticRegression-candidate"
    )


def test_temporal_split_respects_cutoff_dates() -> None:
    """La partición temporal no mezcla observaciones entre train, validation y test."""
    df = pd.DataFrame(
        {
            "hour": pd.to_datetime(
                [
                    "2026-03-31 23:00:00",
                    "2026-04-01 00:00:00",
                    "2026-05-31 23:00:00",
                    "2026-06-01 00:00:00",
                ]
            ),
            "value": [1, 2, 3, 4],
        }
    )

    train, validation, test = temporal_split(df)

    assert train["value"].tolist() == [1]
    assert validation["value"].tolist() == [2, 3]
    assert test["value"].tolist() == [4]


def test_temporal_split_has_no_overlap() -> None:
    """Las tres particiones temporales son disjuntas y cubren todas las observaciones."""
    df = pd.DataFrame(
        {
            "hour": pd.date_range(
                "2026-03-31 20:00:00",
                periods=8,
                freq="h",
            )
        }
    )

    train, validation, test = temporal_split(df)

    train_hours = set(train["hour"])
    validation_hours = set(validation["hour"])
    test_hours = set(test["hour"])

    assert train_hours.isdisjoint(validation_hours)
    assert train_hours.isdisjoint(test_hours)
    assert validation_hours.isdisjoint(test_hours)

    assert len(train) + len(validation) + len(test) == len(df)


def test_create_risk_label_uses_strict_threshold() -> None:
    """El valor exactamente igual al umbral no se clasifica como riesgo."""
    threshold = 0.10

    df = pd.DataFrame(
        {
            "target_imbalance_ratio": [0.05, 0.10, 0.1001, 0.20],
        }
    )

    labels = create_risk_label(df, threshold)

    assert labels.tolist() == [0, 0, 1, 1]


def test_create_risk_label_returns_binary_labels() -> None:
    """La etiqueta de riesgo solo contiene las clases 0 y 1."""
    df = pd.DataFrame(
        {
            "target_imbalance_ratio": [0.01, 0.05, 0.10, 0.20],
        }
    )

    labels = create_risk_label(df, 0.05)

    assert set(labels.unique()) <= {0, 1}
    assert labels.dtype == np.int64


def test_evaluate_model_returns_expected_metrics() -> None:
    """evaluate_model calcula las métricas principales del clasificador."""
    class DummyModel:
        def predict(self, X):
            return np.array([0, 1, 1, 0])

        def predict_proba(self, X):
            return np.array(
                [
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.8, 0.2],
                ]
            )

    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 1, 0])

    metrics = evaluate_model(DummyModel(), X, y, "test")

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)