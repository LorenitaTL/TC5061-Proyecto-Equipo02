"""Pruebas de la configuración (``settings.py``) y de ``GET /health`` (``main.py``)."""

from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

from mi_bici.api.main import create_app, get_predictor
from mi_bici.api.predictor import FixedPredictor, Predictor
from mi_bici.api.settings import Settings
from mi_bici.config import MLFLOW_TRACKING_URI
from mi_bici.modeling.register_p2 import ALIAS, MODEL_NAME

SETTINGS = Settings(mlflow_tracking_uri="sqlite:///:memory:")


def failing_loader(settings: Settings) -> Predictor:
    raise RuntimeError("MLflow no responde")


def client_for(app: FastAPI) -> TestClient:
    return TestClient(app)


# --- settings.py -------------------------------------------------------------------------


def test_from_env_sin_variables_usa_el_modelo_registrado_de_p2():
    settings = Settings.from_env({})

    assert settings == Settings(MLFLOW_TRACKING_URI, MODEL_NAME, ALIAS, None)
    assert settings.model_uri == f"models:/{MODEL_NAME}@{ALIAS}"


def test_from_env_lee_las_variables():
    settings = Settings.from_env(
        {
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MODEL_NAME": "otro-modelo",
            "MODEL_ALIAS": "challenger",
        }
    )

    assert settings.mlflow_tracking_uri == "http://mlflow:5000"
    assert settings.model_uri == "models:/otro-modelo@challenger"


def test_model_version_tiene_prioridad_sobre_el_alias():
    settings = Settings.from_env({"MODEL_ALIAS": "challenger", "MODEL_VERSION": "3"})

    assert settings.model_uri == f"models:/{MODEL_NAME}/3"


@pytest.mark.parametrize("vacio", ["", "  "])
def test_variables_vacias_cuentan_como_no_definidas(vacio):
    settings = Settings.from_env(
        {"MODEL_NAME": vacio, "MODEL_ALIAS": vacio, "MODEL_VERSION": vacio}
    )

    assert settings.model_uri == f"models:/{MODEL_NAME}@{ALIAS}"


def test_from_env_por_omision_lee_os_environ(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "7")

    assert Settings.from_env().model_version == "7"


# --- GET /health -------------------------------------------------------------------------


def test_health_con_modelo_cargado():
    predictor = FixedPredictor(model_name=MODEL_NAME, model_version="3")
    app = create_app(SETTINGS, loader=lambda settings: predictor)

    with client_for(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "model_version": "3",
    }


def test_health_responde_200_aunque_el_modelo_no_cargue():
    app = create_app(SETTINGS, loader=failing_loader)

    with client_for(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": False,
        "model_name": MODEL_NAME,
        "model_version": None,
    }


def test_loader_recibe_la_configuracion():
    received = []

    def loader(settings: Settings) -> Predictor:
        received.append(settings)
        return FixedPredictor()

    with client_for(create_app(SETTINGS, loader=loader)):
        pass

    assert received == [SETTINGS]


def test_sin_settings_la_app_lee_el_entorno(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "modelo-del-entorno")
    app = create_app(loader=failing_loader)

    with client_for(app) as client:
        response = client.get("/health")

    assert response.json()["model_name"] == "modelo-del-entorno"


def test_loader_por_omision_falla_sin_romper_el_arranque():
    # Con sqlite en memoria no hay modelo registrado: load_predictor falla y la API arranca igual.
    with client_for(create_app(SETTINGS)) as client:
        response = client.get("/health")

    assert response.status_code == 200


# --- get_predictor -----------------------------------------------------------------------


def app_with_probe(loader) -> FastAPI:
    app = create_app(SETTINGS, loader=loader)

    @app.get("/probe")
    def probe(predictor: Annotated[Predictor, Depends(get_predictor)]) -> dict:
        return {"model_version": predictor.model_version}

    return app


def test_get_predictor_devuelve_el_modelo_cargado():
    with client_for(app_with_probe(lambda settings: FixedPredictor(model_version="5"))) as c:
        response = c.get("/probe")

    assert response.status_code == 200
    assert response.json() == {"model_version": "5"}


def test_get_predictor_responde_503_sin_modelo():
    with client_for(app_with_probe(failing_loader)) as client:
        response = client.get("/probe")

    assert response.status_code == 503
    assert response.json() == {"detail": "El modelo no está cargado."}


def test_loader_por_omision_usa_load_predictor_de_model_loader(monkeypatch):
    from mi_bici.api import model_loader

    predictor = FixedPredictor(model_version="9")
    monkeypatch.setattr(model_loader, "load_predictor", lambda s: predictor, raising=False)

    with client_for(create_app(SETTINGS)) as client:
        response = client.get("/health")

    assert response.json()["model_version"] == "9"
