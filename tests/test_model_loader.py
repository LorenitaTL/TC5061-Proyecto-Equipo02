from typing import ClassVar

import numpy as np
import pytest

from mi_bici.api import model_loader
from mi_bici.api.predictor import FEATURE_COLUMNS
from mi_bici.api.settings import Settings


class FakeModel:
    def predict_proba(self, features):
        assert list(features.columns) == FEATURE_COLUMNS
        return np.array([[0.8, 0.2]])


class FakeRunData:
    params: ClassVar[dict[str, str]] = {"classification_threshold": "0.35"}
    metrics: ClassVar[dict[str, str]] = {}


class FakeRun:
    data = FakeRunData()


class FakeModelVersion:
    version = "5"
    run_id = "fake-run-id"


class FakeClient:
    def get_model_version_by_alias(self, model_name, alias):
        assert model_name == "mibici-equipo02-p2-riesgo"
        assert alias == "champion"
        return FakeModelVersion()

    def get_run(self, run_id):
        assert run_id == "fake-run-id"
        return FakeRun()


def test_build_features_preserves_p2_contract():
    data = {column: 0.0 for column in FEATURE_COLUMNS}

    features = model_loader.build_features(data)

    assert features.shape == (1, len(FEATURE_COLUMNS))
    assert list(features.columns) == FEATURE_COLUMNS


def test_load_p2_model_resolves_alias_and_threshold(monkeypatch):
    tracking_uri = "sqlite:///temporary-test.db"
    loaded_uris = []
    configured_uris = []

    monkeypatch.setattr(
        model_loader,
        "MLFLOW_TRACKING_URI",
        tracking_uri,
    )
    monkeypatch.setattr(
        model_loader,
        "MlflowClient",
        FakeClient,
    )

    monkeypatch.setattr(
        model_loader.mlflow,
        "set_tracking_uri",
        configured_uris.append,
    )

    class FakeSklearn:
        @staticmethod
        def load_model(uri):
            loaded_uris.append(uri)
            return FakeModel()

    monkeypatch.setattr(
        model_loader.mlflow,
        "sklearn",
        FakeSklearn,
    )

    predictor = model_loader.load_p2_model()

    assert configured_uris == [tracking_uri]
    assert loaded_uris == ["models:/mibici-equipo02-p2-riesgo@champion"]

    assert predictor.model_name == "mibici-equipo02-p2-riesgo"
    assert predictor.model_version == "5"
    assert predictor.threshold == 0.35


def test_loaded_p2_model_predicts_probability_and_class(monkeypatch):
    monkeypatch.setattr(
        model_loader,
        "MLFLOW_TRACKING_URI",
        "sqlite:///temporary-test.db",
    )
    monkeypatch.setattr(
        model_loader,
        "MlflowClient",
        FakeClient,
    )
    monkeypatch.setattr(
        model_loader.mlflow,
        "set_tracking_uri",
        lambda _: None,
    )

    class FakeSklearn:
        @staticmethod
        def load_model(_):
            return FakeModel()

    monkeypatch.setattr(
        model_loader.mlflow,
        "sklearn",
        FakeSklearn,
    )

    predictor = model_loader.load_p2_model()

    data = {column: 0.0 for column in FEATURE_COLUMNS}
    features = model_loader.build_features(data)

    predictions = predictor.predict(features)

    assert len(predictions) == 1
    assert predictions[0].probability == pytest.approx(0.2)
    assert predictions[0].risk_class == 0


class RecordingClient:
    """MlflowClient falso que registra cómo se resolvió la versión."""

    calls: ClassVar[list[tuple]] = []

    def get_model_version_by_alias(self, model_name, alias):
        self.calls.append(("alias", model_name, alias))
        return FakeModelVersion()

    def get_model_version(self, model_name, version):
        self.calls.append(("version", model_name, version))
        return FakeModelVersion()

    def get_run(self, run_id):
        return FakeRun()


def patch_mlflow(monkeypatch) -> tuple[list[str], list[str]]:
    configured_uris: list[str] = []
    loaded_uris: list[str] = []
    RecordingClient.calls = []

    class FakeSklearn:
        @staticmethod
        def load_model(uri):
            loaded_uris.append(uri)
            return FakeModel()

    monkeypatch.setattr(model_loader, "MlflowClient", RecordingClient)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", configured_uris.append)
    monkeypatch.setattr(model_loader.mlflow, "sklearn", FakeSklearn)
    return configured_uris, loaded_uris


def test_load_predictor_usa_la_configuracion_de_la_api(monkeypatch):
    configured_uris, loaded_uris = patch_mlflow(monkeypatch)
    settings = Settings(
        mlflow_tracking_uri="http://mlflow:5000",
        model_name="otro-modelo",
        model_alias="challenger",
    )

    predictor = model_loader.load_predictor(settings)

    assert configured_uris == ["http://mlflow:5000"]
    assert RecordingClient.calls == [("alias", "otro-modelo", "challenger")]
    assert loaded_uris == ["models:/otro-modelo@challenger"]
    assert predictor.model_name == "otro-modelo"
    assert predictor.threshold == 0.35


def test_load_predictor_con_version_fija_ignora_el_alias(monkeypatch):
    _, loaded_uris = patch_mlflow(monkeypatch)
    settings = Settings(mlflow_tracking_uri="sqlite:///x.db", model_version="3")

    model_loader.load_predictor(settings)

    assert RecordingClient.calls == [("version", settings.model_name, "3")]
    assert loaded_uris == [f"models:/{settings.model_name}/3"]
