"""Pruebas del modelo semilla de P2 (L2): datos sintéticos, registro idempotente y carga."""

import json

from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest

from mi_bici.api.model_loader import load_p2_model
from mi_bici.api.predictor import FEATURE_COLUMNS
from mi_bici.api.schemas import PredictionRequest
from mi_bici.config import P2_CLASSIFICATION_THRESHOLD
from mi_bici.operations import seed_model
from mi_bici.operations.seed_model import (
    make_synthetic_features,
    make_synthetic_labels,
    register_seed_model,
)

ROWS = 400


@pytest.fixture
def features() -> pd.DataFrame:
    return make_synthetic_features(ROWS, seed=7)


def test_synthetic_features_follow_the_p2_contract(features):
    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == ROWS
    assert not features.isna().any().any()


def test_synthetic_features_keep_the_relations_of_features_p2(features):
    np.testing.assert_array_equal(
        features["net_flow"], features["departures"] - features["arrivals"]
    )
    np.testing.assert_array_equal(features["absolute_imbalance"], features["net_flow"].abs())
    np.testing.assert_allclose(
        features["imbalance_ratio"],
        features["absolute_imbalance"] / features["capacity"],
        atol=1e-4,
    )
    assert features["is_weekend"].equals((features["day_of_week"] >= 5).astype(int))


def test_synthetic_rows_are_valid_api_requests(features):
    """Cada fila debe poder enviarse tal cual a POST /predict (sirve también para Locust)."""
    rows = json.loads(features.head(20).to_json(orient="records"))

    for row in rows:
        PredictionRequest.model_validate(row)


def test_synthetic_features_are_reproducible():
    pd.testing.assert_frame_equal(
        make_synthetic_features(50, seed=3), make_synthetic_features(50, seed=3)
    )
    assert not make_synthetic_features(50, seed=3).equals(make_synthetic_features(50, seed=4))


def test_synthetic_features_reject_empty_size():
    with pytest.raises(ValueError, match="n_rows"):
        make_synthetic_features(0)


def test_synthetic_labels_have_about_twenty_percent_positives(features):
    labels = make_synthetic_labels(features)

    assert set(np.unique(labels)) == {0, 1}
    assert 0.15 <= labels.mean() <= 0.25


@pytest.fixture
def tracking(tmp_path) -> dict:
    """MLflow local en un directorio temporal: base sqlite y artefactos aparte."""
    return {
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_location": (tmp_path / "artifacts").as_uri(),
    }


def test_register_seed_model_serves_through_the_api_loader(tracking):
    result = register_seed_model(n_rows=ROWS, **tracking)

    predictor = load_p2_model(tracking_uri=tracking["tracking_uri"])
    predictions = predictor.predict(make_synthetic_features(10, seed=11))

    assert result.created is True
    assert predictor.model_version == result.version
    assert predictor.threshold == pytest.approx(P2_CLASSIFICATION_THRESHOLD)
    assert len(predictions) == 10
    assert all(0.0 <= p.probability <= 1.0 for p in predictions)


def test_register_seed_model_marks_the_version_as_synthetic(tracking):
    result = register_seed_model(n_rows=ROWS, **tracking)

    version = MlflowClient(tracking["tracking_uri"]).get_model_version(
        seed_model.MODEL_NAME, result.version
    )

    assert version.tags["origen"] == "seed-sintetico"


def test_register_seed_model_is_idempotent(tracking):
    first = register_seed_model(n_rows=ROWS, **tracking)
    second = register_seed_model(n_rows=ROWS, **tracking)

    versions = MlflowClient(tracking["tracking_uri"]).search_model_versions(
        f"name = '{seed_model.MODEL_NAME}'"
    )

    assert second.created is False
    assert second.version == first.version
    assert len(versions) == 1


def test_register_seed_model_force_moves_the_alias_to_a_new_version(tracking):
    first = register_seed_model(n_rows=ROWS, **tracking)
    forced = register_seed_model(n_rows=ROWS, force=True, **tracking)

    alias_version = seed_model.current_alias_version(
        MlflowClient(tracking["tracking_uri"]), seed_model.MODEL_NAME, seed_model.ALIAS
    )

    assert forced.created is True
    assert forced.version != first.version
    assert alias_version == forced.version
