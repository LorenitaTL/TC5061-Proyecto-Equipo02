from pathlib import Path

import numpy as np
import pandas as pd

from mi_bici.modeling.train_p3 import (
    N_CLUSTERS,
    load_station_coordinates,
    months_between,
    stability_ari,
)


def test_load_station_coordinates_keeps_valid_unique_stations(tmp_path: Path):
    """Descarta coordenadas inválidas e ids duplicados."""
    pd.DataFrame(
        {
            "station_id": pd.array(["1", "1", "2", "3"], dtype="string"),
            "latitude": pd.array([20.67, 20.68, 20.70, None], dtype="Float64"),
            "longitude": pd.array([-103.35, -103.36, -103.40, None], dtype="Float64"),
            "coordinates_valid": pd.array([True, True, True, False], dtype="boolean"),
        }
    ).to_parquet(tmp_path / "estaciones.parquet")

    coordinates = load_station_coordinates(tmp_path)

    assert coordinates.index.tolist() == ["1", "2"]
    assert coordinates.loc["1", "latitude"] == 20.67


def test_months_between_includes_both_ends():
    """Enero a diciembre son 12 meses y enero a junio del año siguiente son 6."""
    assert months_between("2025-01", "2025-12") == 12
    assert months_between("2026-01", "2026-06") == 6


def test_stability_ari_is_one_for_well_separated_groups():
    """Con grupos claramente separados, el resultado no depende de la semilla."""
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]])[:N_CLUSTERS]
    X = np.vstack([center + rng.normal(scale=0.1, size=(20, 2)) for center in centers])

    assert stability_ari(X) == 1
