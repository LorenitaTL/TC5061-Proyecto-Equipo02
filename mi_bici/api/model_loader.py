"""Carga del modelo de P2 desde el Model Registry de MLflow (E1 y E2).

Debe devolver un objeto que cumpla ``Predictor`` (``mi_bici/api/predictor.py``): resuelve
``models:/<MODEL_NAME>@<MODEL_ALIAS>``, lee el umbral ``classification_threshold`` del run
de esa versión y usa ``check_feature_columns`` antes de predecir.
"""

from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
import pandas as pd

from mi_bici.api.predictor import (
    Prediction,
    check_feature_columns,
)
from mi_bici.config import MLFLOW_TRACKING_URI


def build_features(data: dict) -> pd.DataFrame:
    """Convierte los datos de entrada en el DataFrame esperado por P2."""
    features = pd.DataFrame([data])

    check_feature_columns(features)

    return features


@dataclass
class MLflowPredictor:
    """Adaptador del modelo registrado de P2 al contrato Predictor."""

    model: object
    model_name: str
    model_version: str
    threshold: float

    def predict(self, features: pd.DataFrame) -> list[Prediction]:
        """Genera predicciones para las filas recibidas."""
        check_feature_columns(features)

        probabilities = self.model.predict_proba(features)[:, 1]

        return [
            Prediction(
                probability=float(probability),
                risk_class=int(probability >= self.threshold),
            )
            for probability in probabilities
        ]


def load_p2_model(
    model_name: str = "mibici-equipo02-p2-riesgo",
    model_alias: str = "champion",
) -> MLflowPredictor:
    """Carga el modelo P2 desde MLflow Model Registry."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        model_name,
        model_alias,
    )

    model_uri = f"models:/{model_name}@{model_alias}"

    model = mlflow.sklearn.load_model(model_uri)

    run = client.get_run(model_version.run_id)

    threshold_value = run.data.params.get("classification_threshold")

    if threshold_value is None:
        threshold_value = run.data.metrics.get("classification_threshold")

    if threshold_value is None:
        raise ValueError("El run del modelo no contiene 'classification_threshold'.")

    threshold = float(threshold_value)

    return MLflowPredictor(
        model=model,
        model_name=model_name,
        model_version=str(model_version.version),
        threshold=threshold,
    )
