"""Particiones cronológicas del problema P2."""

import pandas as pd

from mi_bici.config import P2_TRAIN_END, P2_VALIDATION_END

TRAIN_END = pd.Timestamp(P2_TRAIN_END)
VALIDATION_END = pd.Timestamp(P2_VALIDATION_END)


def temporal_masks(
    hours: pd.Series,
) -> dict[str, pd.Series]:
    """Devuelve máscaras de entrenamiento, validación y prueba."""
    return {
        "train": hours < TRAIN_END,
        "validation": (hours >= TRAIN_END) & (hours < VALIDATION_END),
        "test": hours >= VALIDATION_END,
    }