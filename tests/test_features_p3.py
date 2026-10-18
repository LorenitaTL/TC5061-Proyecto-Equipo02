from pathlib import Path

import pandas as pd
import pytest

from mi_bici.features_p3 import (
    build_station_profiles,
    flow_balance,
    load_profile_trips,
    load_station_coordinates,
)

MONDAY = "2025-01-06"
SATURDAY = "2025-01-11"


def make_trips(rows: list[tuple[str, str, str, bool]]) -> pd.DataFrame:
    """Crea viajes de 10 minutos a partir de (inicio, origen, destino, duración cero)."""
    started = pd.to_datetime([row[0] for row in rows])
    return pd.DataFrame(
        {
            "started_at": started,
            "ended_at": started + pd.Timedelta(minutes=10),
            "origin_station_id": pd.array([row[1] for row in rows], dtype="string"),
            "destination_station_id": pd.array([row[2] for row in rows], dtype="string"),
            "zero_duration": [row[3] for row in rows],
        }
    )


def commuting_trips() -> pd.DataFrame:
    """Estación A residencial y B laboral: A -> B a las 7 h y B -> A a las 18 h."""
    rows = (
        [(f"{MONDAY} 07:00", "A", "B", False)] * 3
        + [(f"{MONDAY} 18:00", "B", "A", False)] * 3
        + [(f"{SATURDAY} 12:00", "A", "B", False)]
        + [(f"{MONDAY} 07:30", "A", "B", True)] * 5
    )
    return make_trips(rows)


def test_flow_balance_separates_origins_and_destinations():
    """En la mañana A se vacía (balance -1) y B acumula bicis (balance +1)."""
    trips = commuting_trips()
    trips = trips.loc[~trips["zero_duration"]]

    balance = flow_balance(trips, range(6, 10))

    assert balance["A"] == -1
    assert balance["B"] == 1


def test_flow_balance_counts_stations_with_only_arrivals():
    """Una estación sin salidas en la franja tiene balance +1, no se pierde."""
    trips = make_trips([(f"{MONDAY} 07:00", "A", "B", False)])

    balance = flow_balance(trips, range(6, 10))

    assert balance.to_dict() == {"A": -1, "B": 1}


def test_build_station_profiles_variables():
    """Calcula balances entre semana, uso en fin de semana y volumen sin duraciones cero."""
    profiles = build_station_profiles(commuting_trips(), min_departures=1)

    assert profiles.loc["A", "balance_am"] == -1
    assert profiles.loc["A", "balance_pm"] == 1
    assert profiles.loc["A", "balance_dia"] == 0
    # A: 4 salidas válidas (3 el lunes y 1 el sábado); las 5 de duración cero no cuentan.
    assert profiles.loc["A", "share_fin_semana"] == pytest.approx(1 / 4)
    assert profiles.loc["B", "share_fin_semana"] == 0


def test_build_station_profiles_excludes_low_activity_stations():
    """Las estaciones con pocas salidas no tienen un perfil completo y se excluyen."""
    profiles = build_station_profiles(commuting_trips(), min_departures=4)

    assert profiles.index.tolist() == ["A"]


def test_load_profile_trips_reads_month_range(tmp_path: Path):
    """Lee solo los meses del rango pedido."""
    trips = commuting_trips()
    trips.to_parquet(tmp_path / "viajes_2025_01.parquet")
    trips.to_parquet(tmp_path / "viajes_2025_02.parquet")

    loaded = load_profile_trips("2025-01", "2025-01", processed_dir=tmp_path)

    assert len(loaded) == len(trips)


def test_load_profile_trips_fails_on_missing_month(tmp_path: Path):
    """Falla si falta un mes, en lugar de construir perfiles incompletos."""
    commuting_trips().to_parquet(tmp_path / "viajes_2025_01.parquet")

    with pytest.raises(FileNotFoundError, match="viajes_2025_02.parquet"):
        load_profile_trips("2025-01", "2025-02", processed_dir=tmp_path)


def test_load_station_coordinates_keeps_valid_unique_stations(tmp_path: Path):
    """Descarta coordenadas inválidas y conserva la última fila de cada estación."""
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
    assert coordinates.loc["1", "latitude"] == 20.68
