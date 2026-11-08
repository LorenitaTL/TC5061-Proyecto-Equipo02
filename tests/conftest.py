"""Fixtures compartidas de las pruebas de la API (L4 en ``test_api.py``, M4 en
``test_api_validation.py``).

Ninguna necesita MLflow, ``mlflow.db`` ni datos en ``data/``: la app se crea con un
modelo falso (``FixedPredictor`` u otro que cumpla ``Predictor``) o sin modelo.

Este archivo no importa FastAPI ni ``mi_bici.api`` al cargarse: pytest lo carga para
cualquier prueba de ``tests/``, también en el entorno de DeepChecks
(``requirements-validation.txt``), que no instala las dependencias de serving.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from mi_bici.api.predictor import Predictor
    from mi_bici.api.settings import Settings

# Ejemplo del contrato (README, «POST /predict»): estación 10, 13 de noviembre de 2025, 12:00.
VALID_PAYLOAD = {
    "departures": 4.0,
    "arrivals": 3.0,
    "net_flow": 1.0,
    "absolute_imbalance": 1.0,
    "imbalance_ratio": 0.0667,
    "capacity": 15,
    "latitude": 20.6820,
    "longitude": -103.3500,
    "hour_of_day": 12,
    "day_of_week": 3,
    "is_weekend": 0,
    "hour_sin": 0.0,
    "hour_cos": -1.0,
    "dow_sin": 0.4339,
    "dow_cos": -0.9010,
    "departures_lag_1h": 3.0,
    "departures_lag_2h": 2.0,
    "departures_lag_24h": 5.0,
    "departures_rolling_3h": 2.3333,
    "departures_rolling_24h": 1.9167,
    "arrivals_lag_1h": 3.0,
    "arrivals_lag_2h": 1.0,
    "arrivals_lag_24h": 3.0,
    "arrivals_rolling_3h": 1.6667,
    "arrivals_rolling_24h": 2.1667,
    "net_flow_lag_1h": 0.0,
    "net_flow_lag_2h": 1.0,
    "net_flow_lag_24h": 2.0,
    "net_flow_rolling_3h": 0.6667,
    "net_flow_rolling_24h": -0.25,
}


@pytest.fixture
def valid_payload() -> dict:
    """Copia del ejemplo del contrato; cada prueba puede modificarla sin afectar a otras."""

    return dict(VALID_PAYLOAD)


def _no_model(settings: Settings) -> Predictor:
    raise RuntimeError("MLflow no responde")


@pytest.fixture
def api_client() -> Iterator[Callable[[Predictor | None], TestClient]]:
    """Fábrica de clientes: ``api_client(predictor)`` o ``api_client(None)`` para sin modelo.

    El cliente se abre como context manager para que corra el ``lifespan`` (carga del
    modelo) y se cierra al terminar la prueba.
    """

    from fastapi.testclient import TestClient

    from mi_bici.api.main import create_app
    from mi_bici.api.settings import Settings

    settings = Settings(mlflow_tracking_uri="sqlite:///:memory:")

    with ExitStack() as stack:

        def make(predictor: Predictor | None) -> TestClient:
            loader = _no_model if predictor is None else (lambda settings: predictor)
            app = create_app(settings, loader=loader)
            return stack.enter_context(TestClient(app))

        yield make
