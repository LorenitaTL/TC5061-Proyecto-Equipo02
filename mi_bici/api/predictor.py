"""Interfaz entre la API y el modelo de P2.

``POST /predict`` (M3) solo conoce el protocolo ``Predictor``: le pasa un DataFrame con
las variables de ``FEATURE_COLUMNS`` y recibe una ``Prediction`` por fila. El adaptador
real que carga el modelo desde el Model Registry (E1/E2, ``model_loader.py``) implementa
el mismo protocolo, así que el endpoint y sus pruebas pueden usar ``FixedPredictor``
mientras el adaptador no esté listo.
"""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from mi_bici.modeling.train_p2 import FEATURE_COLUMNS

__all__ = [
    "FEATURE_COLUMNS",
    "FixedPredictor",
    "Prediction",
    "Predictor",
    "check_feature_columns",
]


@dataclass(frozen=True)
class Prediction:
    """Resultado del modelo para una estación y hora."""

    probability: float
    """Probabilidad de que la estación tenga desbalance alto en la siguiente hora."""

    risk_class: int
    """1 si ``probability >= threshold``; 0 en otro caso."""


class Predictor(Protocol):
    """Lo que la API necesita de un modelo de riesgo de P2."""

    model_name: str
    model_version: str
    threshold: float

    def predict(self, features: pd.DataFrame) -> list[Prediction]:
        """Devuelve una predicción por fila de ``features``, en el mismo orden."""
        ...


def check_feature_columns(features: pd.DataFrame) -> None:
    """Falla si ``features`` no trae exactamente ``FEATURE_COLUMNS`` en ese orden.

    El modelo se entrenó con ese orden de columnas; un orden distinto no siempre produce
    un error, pero sí predicciones incorrectas.
    """

    columns = list(features.columns)
    if columns != FEATURE_COLUMNS:
        missing = [c for c in FEATURE_COLUMNS if c not in columns]
        extra = [c for c in columns if c not in FEATURE_COLUMNS]
        raise ValueError(
            "Las columnas no coinciden con FEATURE_COLUMNS "
            f"(faltan: {missing}, sobran: {extra}; si ambas están vacías, el orden es distinto)."
        )


@dataclass
class FixedPredictor:
    """Modelo falso que siempre devuelve la misma probabilidad.

    Sirve para desarrollar y probar la API sin MLflow. Valida las columnas igual que el
    adaptador real, para que un error de orden se detecte también en las pruebas.
    """

    probability: float = 0.8
    threshold: float = 0.35
    model_name: str = "modelo-falso"
    model_version: str = "0"

    def predict(self, features: pd.DataFrame) -> list[Prediction]:
        check_feature_columns(features)
        risk_class = int(self.probability >= self.threshold)
        return [Prediction(self.probability, risk_class) for _ in range(len(features))]
