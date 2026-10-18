from pathlib import Path

import pandas as pd
import pytest

from mi_bici.features import (
    add_calendar_features,
    add_station_lags,
    count_station_hourly,
    load_trips,
    month_range,
    trip_files,
)


def test_month_range_includes_both_ends():
    """Enero a diciembre son 12 meses y el formato coincide con los archivos de viajes."""
    months = month_range("2025-01", "2025-12")

    assert len(months) == 12
    assert months[0] == "2025_01"
    assert months[-1] == "2025_12"


def test_trip_files_fails_on_missing_month(tmp_path: Path):
    """Falla si falta un mes, en lugar de usar un periodo incompleto."""
    (tmp_path / "viajes_2025_01.parquet").touch()

    with pytest.raises(FileNotFoundError, match="viajes_2025_02.parquet"):
        trip_files("2025-01", "2025-02", processed_dir=tmp_path)


def test_load_trips_reads_only_requested_columns_and_months(tmp_path: Path):
    """Lee solo las columnas y los meses pedidos."""
    trips = pd.DataFrame({"trip_id": [1, 2], "origin_station_id": ["A", "B"]})
    trips.to_parquet(tmp_path / "viajes_2025_01.parquet")
    trips.to_parquet(tmp_path / "viajes_2025_02.parquet")

    loaded = load_trips(["origin_station_id"], "2025-01", "2025-01", processed_dir=tmp_path)

    assert loaded.columns.tolist() == ["origin_station_id"]
    assert len(loaded) == 2


def test_count_station_hourly_uses_the_hour_of_each_timestamp():
    """Cuenta los viajes de cada estación en la hora en que ocurrieron."""
    trips = pd.DataFrame(
        {
            "origin_station_id": ["A", "A", "B"],
            "started_at": pd.to_datetime(
                ["2025-01-06 07:05", "2025-01-06 07:55", "2025-01-06 08:10"]
            ),
        }
    )

    counts = count_station_hourly(trips, "origin_station_id", "started_at", "departures")

    assert counts.columns.tolist() == ["station_id", "hour", "departures"]
    assert counts["departures"].tolist() == [2, 1]
    assert counts["hour"].dt.hour.tolist() == [7, 8]


def test_add_calendar_features():
    """El viernes es laboral y el sábado es fin de semana."""
    df = pd.DataFrame({"hour": pd.to_datetime(["2026-06-05 08:00", "2026-06-06 14:00"])})

    result = add_calendar_features(df)

    assert result["hour_of_day"].tolist() == [8, 14]
    assert result["day_of_week"].tolist() == [4, 5]
    assert result["is_weekend"].tolist() == [0, 1]


def test_add_station_lags_sorts_before_shifting():
    """Los rezagos siguen el tiempo de cada estación aunque las filas lleguen desordenadas."""
    hours = pd.date_range("2026-06-01", periods=3, freq="h")
    df = pd.DataFrame(
        {
            "station_id": ["B", "A", "B", "A", "B", "A"],
            "hour": [hours[2], hours[1], hours[0], hours[0], hours[1], hours[2]],
            "departures": [12, 2, 10, 1, 11, 3],
        }
    )

    result = add_station_lags(df, "departures", [1], prefix="departures_")

    a = result.loc[result["station_id"] == "A"]
    b = result.loc[result["station_id"] == "B"]
    assert a["departures_lag_1h"].tolist()[1:] == [1, 2]
    assert b["departures_lag_1h"].tolist()[1:] == [10, 11]
    assert result["departures_lag_1h"].isna().sum() == 2
