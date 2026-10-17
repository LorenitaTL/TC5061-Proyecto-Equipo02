
"""Pruebas de construcción de variables para P1."""

import pandas as pd

from mi_bici.features_p1 import add_features, build_hourly_grid


def test_hourly_grid_fills_missing_hours() -> None:
    """Las horas sin viajes se representan con cero."""
    hourly = pd.DataFrame(
        {
            "station_id": ["10", "10"],
            "hour": pd.to_datetime(
                ["2026-06-01 08:00", "2026-06-01 10:00"]
            ),
            "trip_count": [3, 5],
        }
    )

    result = build_hourly_grid(hourly)

    assert len(result) == 3
    assert result["trip_count"].tolist() == [3, 0, 5]
    assert result["trip_count"].sum() == 8


def test_hourly_grid_preserves_station_identity() -> None:
    """Los conteos de distintas estaciones no se mezclan."""
    hourly = pd.DataFrame(
        {
            "station_id": ["10", "20"],
            "hour": pd.to_datetime(
                ["2026-06-01 08:00", "2026-06-01 08:00"]
            ),
            "trip_count": [3, 7],
        }
    )

    result = build_hourly_grid(hourly)

    assert len(result) == 2
    assert result["trip_count"].sum() == 10
    assert result["station_id"].nunique() == 2


def test_lags_respect_station_and_time() -> None:
    """Los rezagos corresponden a la historia de cada estación."""
    hours = pd.date_range(
        "2026-06-01",
        periods=170,
        freq="h",
    )

    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "station_id": station,
                    "hour": hours,
                    "trip_count": range(offset, offset + 170),
                }
            )
            for station, offset in [("10", 0), ("20", 1000)]
        ],
        ignore_index=True,
    )

    result = add_features(df)

    for station, offset in [("10", 0), ("20", 1000)]:
        subset = result.loc[
            result["station_id"] == station
        ].reset_index(drop=True)

        assert subset.loc[168, "lag_1h"] == offset + 167
        assert subset.loc[168, "lag_24h"] == offset + 144
        assert subset.loc[168, "lag_168h"] == offset

    assert result["lag_168h"].isna().sum() == 336


def test_calendar_features() -> None:
    """Las variables calendario se calculan correctamente."""
    df = pd.DataFrame(
        {
            "station_id": ["10", "10"],
            "hour": pd.to_datetime(
                ["2026-06-05 08:00", "2026-06-06 14:00"]
            ),
            "trip_count": [4, 2],
        }
    )

    result = add_features(df)

    assert result["hour_of_day"].tolist() == [8, 14]
    assert result["day_of_week"].tolist() == [4, 5]
    assert result["is_weekend"].tolist() == [0, 1]
