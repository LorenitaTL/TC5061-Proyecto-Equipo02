import asyncio

import httpx
import pytest

from mi_bici.api.main import create_app
from mi_bici.api.predictor import FixedPredictor
from mi_bici.api.schemas import PredictionRequest


def post_prediction(payload: dict) -> httpx.Response:
    async def send_request() -> httpx.Response:
        predictor = FixedPredictor(probability=0.35, threshold=0.35)
        app = create_app(loader=lambda _: predictor)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post("/predict", json=payload)

    return asyncio.run(send_request())


def test_predict_devuelve_la_respuesta_del_contrato():
    payload = PredictionRequest.model_json_schema()["examples"][0]

    response = post_prediction(payload)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "risk_probability": 0.35,
        "risk_class": 1,
        "risk_label": "alto",
        "threshold": 0.35,
        "model_name": "modelo-falso",
        "model_version": "0",
    }


@pytest.mark.parametrize("invalid_hour", ["12", 24], ids=["tipo-incorrecto", "fuera-de-rango"])
def test_predict_rechaza_una_hora_invalida(invalid_hour):
    payload = PredictionRequest.model_json_schema()["examples"][0]

    response = post_prediction({**payload, "hour_of_day": invalid_hour})

    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["body", "hour_of_day"]


def test_predict_rechaza_un_campo_obligatorio_faltante():
    payload = PredictionRequest.model_json_schema()["examples"][0].copy()
    del payload["capacity"]

    response = post_prediction(payload)

    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["body", "capacity"]
