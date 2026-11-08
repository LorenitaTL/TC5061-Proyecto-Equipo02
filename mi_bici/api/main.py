"""Aplicación FastAPI: ``GET /health`` (L3) y ``POST /predict`` (M3).

Se arranca con ``uvicorn mi_bici.api.main:app``. Al arrancar se lee la configuración
(``settings.py``) y se intenta cargar el modelo con ``load_predictor`` de
``model_loader.py``. Si la carga falla, la API arranca igual: ``/health`` responde
``model_loaded: false`` y los endpoints que usan ``get_predictor`` responden 503.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from loguru import logger
import pandas as pd
from pydantic import BaseModel

from mi_bici.api.predictor import FEATURE_COLUMNS, Predictor
from mi_bici.api.schemas import PredictionRequest, PredictionResponse
from mi_bici.api.settings import Settings

PredictorLoader = Callable[[Settings], Predictor]
"""Recibe la configuración y devuelve el modelo listo para predecir, o lanza una excepción."""


def load_registered_predictor(settings: Settings) -> Predictor:
    """Carga el modelo del Model Registry con el adaptador de ``model_loader.py`` (E1/E2)."""

    from mi_bici.api.model_loader import load_predictor

    return load_predictor(settings)


class HealthResponse(BaseModel):
    """Respuesta de ``GET /health``."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str | None


def create_app(
    settings: Settings | None = None,
    loader: PredictorLoader = load_registered_predictor,
) -> FastAPI:
    """Crea la app. Las pruebas pasan su propia configuración y un ``loader`` falso."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings or Settings.from_env()
        app.state.predictor = None
        try:
            app.state.predictor = loader(app.state.settings)
        # Cualquier fallo (MLflow caído, alias inexistente, adaptador sin implementar) deja
        # la API sin modelo en vez de impedir el arranque; /health lo reporta.
        except Exception:  # noqa: BLE001
            logger.exception(
                f"No se pudo cargar el modelo {app.state.settings.model_uri}; "
                "la API arranca sin modelo."
            )
        else:
            logger.info(
                f"Modelo cargado: {app.state.predictor.model_name} "
                f"versión {app.state.predictor.model_version}."
            )
        yield

    app = FastAPI(
        title="MiBici: riesgo de desbalance (P2)",
        summary="Predice si una estación tendrá desbalance alto en la siguiente hora.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        predictor: Predictor | None = request.app.state.predictor
        if predictor is None:
            return HealthResponse(
                status="ok",
                model_loaded=False,
                model_name=request.app.state.settings.model_name,
                model_version=None,
            )
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=predictor.model_name,
            model_version=predictor.model_version,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        predictor: Annotated[Predictor, Depends(get_predictor)],
    ) -> PredictionResponse:
        features = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
        prediction = predictor.predict(features)[0]

        return PredictionResponse(
            risk_probability=prediction.probability,
            risk_class=prediction.risk_class,
            risk_label="alto" if prediction.risk_class == 1 else "bajo",
            threshold=predictor.threshold,
            model_name=predictor.model_name,
            model_version=predictor.model_version,
        )

    return app


def get_predictor(request: Request) -> Predictor:
    """Dependencia para ``POST /predict``: el modelo cargado, o 503 si no lo hay."""

    predictor: Predictor | None = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está cargado.",
        )
    return predictor


app = create_app()
