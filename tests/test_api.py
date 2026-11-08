"""Pruebas de integración de la API: flujo HTTP completo de ``/health`` y ``/predict``.

Usan modelos falsos, así que no necesitan MLflow. Las reglas de validación de cada campo
se prueban en ``test_api_validation.py`` (M4); aquí solo el flujo de punta a punta y los
códigos HTTP del contrato (README, «Contrato de la API»).
"""

from dataclasses import dataclass, field
import random

import pandas as pd

from mi_bici.api.predictor import FEATURE_COLUMNS, FixedPredictor, Prediction

MODEL_NAME = "mibici-equipo02-p2-riesgo"


@dataclass
class RecordingPredictor:
    """Modelo falso que guarda los DataFrames que recibe."""

    probability: float = 0.42
    threshold: float = 0.35
    model_name: str = MODEL_NAME
    model_version: str = "3"
    received: list[pd.DataFrame] = field(default_factory=list)

    def predict(self, features: pd.DataFrame) -> list[Prediction]:
        self.received.append(features.copy())
        risk_class = int(self.probability >= self.threshold)
        return [Prediction(self.probability, risk_class) for _ in range(len(features))]


# --- Flujo completo ----------------------------------------------------------------------


def test_health_y_predict_reportan_el_mismo_modelo(api_client, valid_payload):
    client = api_client(FixedPredictor(model_name=MODEL_NAME, model_version="3"))

    health = client.get("/health").json()
    prediction = client.post("/predict", json=valid_payload).json()

    assert health["model_loaded"] is True
    assert (health["model_name"], health["model_version"]) == (MODEL_NAME, "3")
    assert (prediction["model_name"], prediction["model_version"]) == (MODEL_NAME, "3")


def test_predict_valido_responde_con_el_contrato(api_client, valid_payload):
    client = api_client(RecordingPredictor(probability=0.42, threshold=0.35))

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json() == {
        "risk_probability": 0.42,
        "risk_class": 1,
        "risk_label": "alto",
        "threshold": 0.35,
        "model_name": MODEL_NAME,
        "model_version": "3",
    }


def test_predict_bajo_el_umbral_es_riesgo_bajo(api_client, valid_payload):
    client = api_client(RecordingPredictor(probability=0.10, threshold=0.35))

    body = client.post("/predict", json=valid_payload).json()

    assert (body["risk_class"], body["risk_label"]) == (0, "bajo")


def test_predict_pasa_al_modelo_una_fila_en_el_orden_del_entrenamiento(api_client, valid_payload):
    predictor = RecordingPredictor()
    client = api_client(predictor)
    keys = list(valid_payload)
    random.Random(0).shuffle(keys)
    shuffled = {key: valid_payload[key] for key in keys}

    response = client.post("/predict", json=shuffled)

    assert response.status_code == 200
    [features] = predictor.received
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.iloc[0].to_dict() == valid_payload


# --- Errores -----------------------------------------------------------------------------


def test_predict_sin_una_variable_responde_422(api_client, valid_payload):
    predictor = RecordingPredictor()
    client = api_client(predictor)
    del valid_payload["capacity"]

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert predictor.received == []


def test_predict_con_una_variable_de_mas_responde_422(api_client, valid_payload):
    predictor = RecordingPredictor()
    client = api_client(predictor)
    valid_payload["station_id"] = 10

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert predictor.received == []


def test_predict_con_un_valor_no_numerico_responde_422(api_client, valid_payload):
    predictor = RecordingPredictor()
    client = api_client(predictor)
    valid_payload["departures"] = "cuatro"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert predictor.received == []


def test_predict_sin_modelo_responde_503(api_client, valid_payload):
    client = api_client(None)

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 503
    assert client.get("/health").json()["model_loaded"] is False
