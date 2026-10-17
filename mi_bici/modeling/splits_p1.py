"""Particiones cronológicas del problema P1."""

import pandas as pd

TRAIN_END = pd.Timestamp("2026-04-01")
VALIDATION_END = pd.Timestamp("2026-06-01")


def temporal_masks(
    hours: pd.Series,
) -> dict[str, pd.Series]:
    """Devuelve máscaras de entrenamiento, validación y prueba."""
    return {
        "train": hours < TRAIN_END,
        "validation": (hours >= TRAIN_END) & (hours < VALIDATION_END),
        "test": hours >= VALIDATION_END,
    }
