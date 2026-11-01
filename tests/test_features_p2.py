"""Pruebas de la construcción de features de P2."""

import numpy as np
import pandas as pd
import pytest

from mi_bici.features_p2 import (
    add_historical_features,
    add_imbalance_features,
    add_next_hour_target,
    add_temporal_features,
    aggregate_station_flows,
    create_station_hour_grid,
)


def make_trips(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Crea viajes a partir de (inicio, fin, estación de origen, estación de destino)."""
    return pd.DataFrame(
        {
            "started_at": pd.to_datetime([row[0] for row in rows]),
            "ended_at": pd.to_datetime([row[1] for row in rows]),
            "origin_station_id": pd.array([row[2] for row in rows], dtype="string"),
            "destination_station_id": pd.array([row[3] for row in rows], dtype="string"),
        }
    )


def make_stations(capacities: dict[str, int]) -> pd.DataFrame:
    """Crea un catálogo de estaciones con su capacidad."""
    ids = list(capacities)
    return pd.DataFrame(
        {
            "station_id": pd.array(ids, dtype="string"),
            "station_name": [f"Estación {station}" for station in ids],
            "capacity": list(capacities.values()),
            "latitude": 20.67,
            "longitude": -103.35,
        }
    )


def build_grid(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Agrega los viajes y arma la cuadrícula, como en build_station_hour_table."""
    departures, arrivals = aggregate_station_flows(trips)
    return create_station_hour_grid(departures, arrivals, stations)


def row(grid: pd.DataFrame, station: str, hour: str) -> pd.Series:
    """Devuelve la única fila de una estación en una hora."""
    match = grid[(grid["station_id"] == station) & (grid["hour"] == pd.Timestamp(hour))]
    assert len(match) == 1
    return match.iloc[0]


# --- C9: aggregate_station_flows ------------------------------------------------------


def test_departures_use_start_and_arrivals_use_end() -> None:
    """Un viaje que cruza de hora sale en la hora de inicio y llega en la hora de fin."""
    trips = make_trips([("2025-01-06 08:50", "2025-01-06 09:10", "A", "B")])
    departures, arrivals = aggregate_station_flows(trips)

    assert departures.to_dict("records") == [
        {"station_id": "A", "hour": pd.Timestamp("2025-01-06 08:00"), "departures": 1}
    ]
    assert arrivals.to_dict("records") == [
        {"station_id": "B", "hour": pd.Timestamp("2025-01-06 09:00"), "arrivals": 1}
    ]


def test_trips_are_counted_per_station_and_hour() -> None:
    """Los viajes de la misma estación y hora se suman; el minuto 00 abre una hora nueva."""
    trips = make_trips(
        [
            ("2025-01-06 08:00", "2025-01-06 08:10", "A", "B"),
            ("2025-01-06 08:59", "2025-01-06 09:05", "A", "B"),
            ("2025-01-06 09:00", "2025-01-06 09:20", "A", "C"),
        ]
    )
    departures, _ = aggregate_station_flows(trips)

    counts = departures.set_index("hour")["departures"].to_dict()
    assert counts == {pd.Timestamp("2025-01-06 08:00"): 2, pd.Timestamp("2025-01-06 09:00"): 1}


def test_round_trip_counts_as_departure_and_arrival() -> None:
    """Un viaje que regresa a su estación cuenta como salida y como llegada."""
    trips = make_trips([("2025-01-06 08:10", "2025-01-06 08:40", "A", "A")])
    departures, arrivals = aggregate_station_flows(trips)

    assert departures["departures"].tolist() == [1]
    assert arrivals["arrivals"].tolist() == [1]


# --- C9: create_station_hour_grid -----------------------------------------------------


def test_grid_has_one_row_per_station_and_hour() -> None:
    """La cuadrícula es estaciones × horas, sin duplicados."""
    trips = make_trips(
        [
            ("2025-01-06 08:10", "2025-01-06 08:20", "A", "B"),
            ("2025-01-06 10:10", "2025-01-06 10:20", "B", "A"),
        ]
    )
    grid = build_grid(trips, make_stations({"A": 10, "B": 20, "C": 30}))

    assert len(grid) == 3 * 3  # 3 estaciones × horas 08, 09 y 10
    assert not grid.duplicated(["station_id", "hour"]).any()


def test_station_without_trips_has_zero_flow() -> None:
    """Una estación del catálogo sin viajes aparece en todas las horas con flujo cero."""
    trips = make_trips(
        [
            ("2025-01-06 08:10", "2025-01-06 08:20", "A", "B"),
            ("2025-01-06 09:10", "2025-01-06 09:20", "B", "A"),
        ]
    )
    grid = build_grid(trips, make_stations({"A": 10, "B": 20, "C": 30}))

    station_c = grid[grid["station_id"] == "C"]
    assert len(station_c) == 2
    assert (station_c["departures"] == 0).all()
    assert (station_c["arrivals"] == 0).all()


def test_hour_without_flow_is_filled_with_zero() -> None:
    """Una hora intermedia sin viajes existe en la cuadrícula con flujo cero."""
    trips = make_trips(
        [
            ("2025-01-06 08:10", "2025-01-06 08:20", "A", "B"),
            ("2025-01-06 10:10", "2025-01-06 10:20", "A", "B"),
        ]
    )
    grid = build_grid(trips, make_stations({"A": 10, "B": 20}))

    empty_hour = row(grid, "A", "2025-01-06 09:00")
    assert empty_hour["departures"] == 0
    assert empty_hour["arrivals"] == 0
    assert row(grid, "A", "2025-01-06 10:00")["departures"] == 1


def test_grid_covers_arrivals_after_last_departure() -> None:
    """El rango de horas llega hasta la última llegada, aunque sea posterior a toda salida."""
    trips = make_trips([("2025-01-06 08:50", "2025-01-06 10:05", "A", "B")])
    grid = build_grid(trips, make_stations({"A": 10, "B": 20}))

    assert grid["hour"].max() == pd.Timestamp("2025-01-06 10:00")
    assert row(grid, "B", "2025-01-06 10:00")["arrivals"] == 1


def test_grid_keeps_station_attributes() -> None:
    """Cada fila trae la capacidad de su estación, necesaria para el desbalance."""
    trips = make_trips([("2025-01-06 08:10", "2025-01-06 08:20", "A", "B")])
    grid = build_grid(trips, make_stations({"A": 10, "B": 20}))

    capacities = grid.groupby("station_id")["capacity"].unique()
    assert capacities["A"].tolist() == [10]
    assert capacities["B"].tolist() == [20]


def test_grid_drops_stations_missing_from_catalog() -> None:
    """Los viajes de estaciones fuera del catálogo no generan filas."""
    trips = make_trips([("2025-01-06 08:10", "2025-01-06 08:20", "A", "X")])
    grid = build_grid(trips, make_stations({"A": 10}))

    assert grid["station_id"].unique().tolist() == ["A"]


# --- C10: helpers ---------------------------------------------------------------------


def make_hourly(
    departures: dict[str, list[float]],
    arrivals: dict[str, list[float]] | None = None,
    capacity: float = 10,
    start: str = "2025-01-06 00:00",
) -> pd.DataFrame:
    """Crea filas estación-hora consecutivas con su flujo, como la cuadrícula de P2."""
    frames = []
    for station, values in departures.items():
        incoming = arrivals[station] if arrivals else [0.0] * len(values)
        frames.append(
            pd.DataFrame(
                {
                    "station_id": station,
                    "hour": pd.date_range(start, periods=len(values), freq="h"),
                    "departures": [float(value) for value in values],
                    "arrivals": [float(value) for value in incoming],
                    "capacity": capacity,
                }
            )
        )
    hourly = pd.concat(frames, ignore_index=True)
    hourly["net_flow"] = hourly["departures"] - hourly["arrivals"]
    return hourly


# --- C10: add_imbalance_features ------------------------------------------------------


def test_imbalance_sign_magnitude_and_ratio() -> None:
    """net_flow conserva el signo; el desbalance absoluto se normaliza por capacidad."""
    hourly = pd.DataFrame(
        {
            "departures": [5.0, 1.0, 4.0],
            "arrivals": [2.0, 4.0, 4.0],
            "capacity": [10, 20, 10],
        }
    )
    result = add_imbalance_features(hourly)

    assert result["net_flow"].tolist() == [3.0, -3.0, 0.0]
    assert result["absolute_imbalance"].tolist() == [3.0, 3.0, 0.0]
    assert result["imbalance_ratio"].tolist() == pytest.approx([0.3, 0.15, 0.0])


def test_imbalance_ratio_compares_station_sizes() -> None:
    """El mismo desbalance pesa más en una estación pequeña."""
    hourly = pd.DataFrame(
        {"departures": [10.0, 10.0], "arrivals": [0.0, 0.0], "capacity": [20, 80]}
    )
    ratios = add_imbalance_features(hourly)["imbalance_ratio"].tolist()

    assert ratios == pytest.approx([0.5, 0.125])


# --- C10: add_temporal_features -------------------------------------------------------


def test_temporal_features_from_timestamp() -> None:
    """Hora, día y fin de semana salen del timestamp (6 ene 2025 es lunes)."""
    hourly = pd.DataFrame(
        {"hour": pd.to_datetime(["2025-01-06 08:00", "2025-01-11 23:00", "2025-01-12 00:00"])}
    )
    result = add_temporal_features(hourly)

    assert result["hour_of_day"].tolist() == [8, 23, 0]
    assert result["day_of_week"].tolist() == [0, 5, 6]
    assert result["is_weekend"].tolist() == [0, 1, 1]


def test_cyclic_hour_keeps_midnight_close_to_23h() -> None:
    """En la codificación cíclica, 23:00 queda cerca de 00:00 y lejos de 12:00."""
    hourly = pd.DataFrame(
        {"hour": pd.to_datetime(["2025-01-06 00:00", "2025-01-06 23:00", "2025-01-06 12:00"])}
    )
    result = add_temporal_features(hourly)
    points = result[["hour_sin", "hour_cos"]].to_numpy()

    assert points[0] == pytest.approx([0.0, 1.0])
    assert np.linalg.norm(points[0] - points[1]) < np.linalg.norm(points[0] - points[2])


# --- C10: add_historical_features -----------------------------------------------------


def test_first_hour_has_no_history() -> None:
    """La primera hora de una estación no tiene rezagos ni promedios móviles."""
    result = add_historical_features(make_hourly({"A": list(range(30))}))
    first = result.iloc[0]

    history = [column for column in result.columns if "_lag_" in column or "_rolling_" in column]
    assert len(history) == 15  # 3 columnas × (3 rezagos + 2 ventanas)
    assert first[history].isna().all()


def test_lags_look_only_at_previous_hours() -> None:
    """lag_nh es el valor n horas antes; con departures = 0, 1, 2, ... vale t - n."""
    result = add_historical_features(make_hourly({"A": list(range(30))}))

    at_25h = result.iloc[25]
    assert at_25h["departures_lag_1h"] == 24
    assert at_25h["departures_lag_2h"] == 23
    assert at_25h["departures_lag_24h"] == 1
    assert pd.isna(result.iloc[1]["departures_lag_2h"])
    assert pd.isna(result.iloc[23]["departures_lag_24h"])


def test_rolling_windows_exclude_current_hour() -> None:
    """Las ventanas promedian las horas anteriores, nunca la hora actual."""
    result = add_historical_features(make_hourly({"A": list(range(30))}))

    assert result.iloc[1]["departures_rolling_3h"] == 0  # solo t-1
    assert result.iloc[2]["departures_rolling_3h"] == 0.5  # promedio de 0 y 1
    assert result.iloc[10]["departures_rolling_3h"] == 8  # promedio de 7, 8 y 9
    assert result.iloc[24]["departures_rolling_24h"] == 11.5  # promedio de 0 a 23


def test_history_does_not_mix_stations() -> None:
    """La primera hora de la estación B no hereda la historia de la estación A."""
    hourly = make_hourly({"A": [5, 5, 5], "B": [1, 2, 3]})
    result = add_historical_features(hourly)
    station_b = result[result["station_id"] == "B"].reset_index(drop=True)

    assert pd.isna(station_b.loc[0, "departures_lag_1h"])
    assert pd.isna(station_b.loc[0, "departures_rolling_3h"])
    assert station_b.loc[1, "departures_lag_1h"] == 1


# --- C10: add_next_hour_target --------------------------------------------------------


def test_target_is_next_hour_of_same_station() -> None:
    """El objetivo de t es el flujo y el desbalance de t+1 en la misma estación."""
    hourly = make_hourly({"A": [1, 6, 2]}, arrivals={"A": [0, 1, 7]}, capacity=10)
    result = add_next_hour_target(hourly)

    first = result.iloc[0]
    assert first["target_departures"] == 6
    assert first["target_arrivals"] == 1
    assert first["target_net_flow"] == 5
    assert first["target_imbalance_ratio"] == pytest.approx(0.5)
    assert result.iloc[1]["target_imbalance_ratio"] == pytest.approx(0.5)  # |2 - 7| / 10


def test_last_hour_without_target_is_dropped() -> None:
    """La última hora de cada estación no tiene t+1, así que se elimina."""
    hourly = make_hourly({"A": [1, 2, 3], "B": [4, 5]})
    result = add_next_hour_target(hourly)

    assert result.groupby("station_id").size().to_dict() == {"A": 2, "B": 1}
    assert result["hour"].max() < hourly["hour"].max()
    assert result["target_imbalance_ratio"].notna().all()


def test_target_does_not_cross_stations_or_depend_on_row_order() -> None:
    """Aunque lleguen desordenadas, B no toma como objetivo la primera hora de otra estación."""
    hourly = make_hourly({"A": [1, 2], "B": [7, 9]}).sample(frac=1, random_state=0)
    result = add_next_hour_target(hourly)

    targets = result.set_index("station_id")["target_departures"].to_dict()
    assert targets == {"A": 2, "B": 9}
