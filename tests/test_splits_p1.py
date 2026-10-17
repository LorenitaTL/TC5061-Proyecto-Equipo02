"""Pruebas de las particiones temporales de P1."""

import pandas as pd

from mi_bici.modeling.splits_p1 import temporal_masks


def test_partition_boundaries() -> None:
    """Cada fecha pertenece al periodo esperado."""
    hours = pd.Series(
        pd.to_datetime(
            [
                "2026-03-31 23:00",
                "2026-04-01 00:00",
                "2026-05-31 23:00",
                "2026-06-01 00:00",
            ]
        )
    )

    masks = temporal_masks(hours)

    assert masks["train"].tolist() == [True, False, False, False]
    assert masks["validation"].tolist() == [False, True, True, False]
    assert masks["test"].tolist() == [False, False, False, True]


def test_partitions_are_exclusive_and_complete() -> None:
    """Ninguna observación pertenece a dos conjuntos."""
    hours = pd.Series(pd.date_range("2026-03-30", periods=100, freq="D"))

    masks = temporal_masks(hours)

    membership = (
        masks["train"].astype(int) + masks["validation"].astype(int) + masks["test"].astype(int)
    )

    assert (membership == 1).all()
