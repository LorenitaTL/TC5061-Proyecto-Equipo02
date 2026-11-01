"""Pruebas de la construcción de features de P2."""

import pandas as pd

from mi_bici.features_p2 import aggregate_station_flows, create_station_hour_grid


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
