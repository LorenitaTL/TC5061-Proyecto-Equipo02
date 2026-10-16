
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from mi_bici.dataset import (
    calculate_sha256,
    normalize_stations,
    normalize_trips,
)

def test_normalize_stations_flags_invalid_capacity():
    """Identifica estaciones con capacidad no válida."""
    raw = pd.DataFrame(
        {
            "id": ["1", "2"],
            "name": ["Estación A", "Estación B"],
            "latitude": ["20.67", "20.68"],
            "longitude": ["-103.35", "-103.36"],
            "dpcapacity": ["20", "0"],
        }
    )

    stations = normalize_stations(raw)

    assert len(stations) == 2
    assert stations["capacity_valid"].tolist() == [True, False]
    assert stations["coordinates_valid"].all()

def test_calculate_sha256(tmp_path: Path):
    """El hash debe coincidir con un valor conocido."""
    file_path = tmp_path / "example.txt"
    content = b"MiBici MLOps"

    file_path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()

    assert calculate_sha256(file_path) == expected


def test_normalize_trips_preserves_rows():
    """La normalización no debe descartar registros."""
    raw = pd.DataFrame(
        {
            "Viaje_Id": ["1", "2"],
            "Usuario_Id": ["10", "11"],
            "Genero": ["M", "F"],
            "Año_de_nacimiento": ["1990", "1995"],
            "Inicio_del_viaje": [
                "2026-01-01 08:00:00",
                "2026-01-01 09:00:00",
            ],
            "Fin_del_viaje": [
                "2026-01-01 08:10:00",
                "2026-01-01 09:00:00",
            ],
            "Origen_Id": ["35", "51"],
            "Destino_Id": ["51", "51"],
        }
    )

    normalized = normalize_trips(raw)

    assert len(normalized) == len(raw)
    assert normalized["trip_id"].is_unique
    assert normalized["zero_duration"].sum() == 1


def test_normalize_trips_flags_negative_duration():
    """Una duración negativa debe quedar identificada."""
    raw = pd.DataFrame(
        {
            "Viaje_Id": ["3"],
            "Inicio_del_viaje": ["2026-01-01 10:00:00"],
            "Fin_del_viaje": ["2026-01-01 09:50:00"],
            "Origen_Id": ["35"],
            "Destino_Id": ["51"],
        }
    )

    normalized = normalize_trips(raw)

    assert bool(normalized.loc[0, "negative_duration"])
    assert len(normalized) == 1
