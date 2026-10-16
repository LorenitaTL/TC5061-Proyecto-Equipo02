from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from mi_bici.config import PROCESSED_DATA_DIR

app = typer.Typer()


def load_trips(processed_dir: Path) -> pd.DataFrame:
    """Carga todos los Parquet mensuales de viajes MiBici."""
    files = sorted(processed_dir.glob("viajes_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos viajes_*.parquet en {processed_dir}"
        )

    frames = []
    for path in files:
        logger.info(f"Leyendo {path.name}")
        frames.append(
            pd.read_parquet(
                path,
                columns=[
                    "trip_id",
                    "started_at",
                    "ended_at",
                    "origin_station_id",
                    "destination_station_id",
                ],
            )
        )

    trips = pd.concat(frames, ignore_index=True)
    trips["started_at"] = pd.to_datetime(trips["started_at"], errors="coerce")
    trips["ended_at"] = pd.to_datetime(trips["ended_at"], errors="coerce")

    trips = trips.dropna(
        subset=[
            "started_at",
            "ended_at",
            "origin_station_id",
            "destination_station_id",
        ]
    ).copy()

    for col in ["origin_station_id", "destination_station_id"]:
        trips[col] = trips[col].astype("string").str.strip()

    logger.success(f"Viajes cargados: {len(trips):,}")
    return trips


def load_stations(processed_dir: Path) -> pd.DataFrame:
    """Carga el catálogo normalizado de estaciones."""
    path = processed_dir / "estaciones.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró {path}")

    stations = pd.read_parquet(
        path,
        columns=[
            "station_id",
            "station_name",
            "capacity",
            "latitude",
            "longitude",
        ],
    )

    stations["station_id"] = (
        stations["station_id"].astype("string").str.strip()
    )
    stations["capacity"] = pd.to_numeric(
        stations["capacity"], errors="coerce"
    )

    stations = stations.drop_duplicates(
        subset=["station_id"], keep="last"
    )
    stations = stations[stations["capacity"] > 0].copy()

    logger.success(f"Estaciones válidas: {len(stations):,}")
    return stations


def build_station_hour_table(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """Construye una observación por estación-hora para P2."""

    departures = (
        trips.assign(hour=trips["started_at"].dt.floor("h"))
        .groupby(["origin_station_id", "hour"], as_index=False)
        .size()
        .rename(
            columns={
                "origin_station_id": "station_id",
                "size": "departures",
            }
        )
    )

    arrivals = (
        trips.assign(hour=trips["ended_at"].dt.floor("h"))
        .groupby(["destination_station_id", "hour"], as_index=False)
        .size()
        .rename(
            columns={
                "destination_station_id": "station_id",
                "size": "arrivals",
            }
        )
    )

    min_hour = min(departures["hour"].min(), arrivals["hour"].min())
    max_hour = max(departures["hour"].max(), arrivals["hour"].max())
    hours = pd.date_range(min_hour, max_hour, freq="h")

    full_grid = (
        stations[["station_id"]]
        .assign(key=1)
        .merge(
            pd.DataFrame({"hour": hours, "key": 1}),
            on="key",
        )
        .drop(columns="key")
    )

    hourly = full_grid.merge(
        departures, on=["station_id", "hour"], how="left"
    )
    hourly = hourly.merge(
        arrivals, on=["station_id", "hour"], how="left"
    )

    hourly["departures"] = hourly["departures"].fillna(0)
    hourly["arrivals"] = hourly["arrivals"].fillna(0)

    hourly = hourly.merge(
        stations,
        on="station_id",
        how="left",
        validate="many_to_one",
    )

    hourly["net_flow"] = (
        hourly["departures"] - hourly["arrivals"]
    )
    hourly["absolute_imbalance"] = hourly["net_flow"].abs()
    hourly["imbalance_ratio"] = (
        hourly["absolute_imbalance"] / hourly["capacity"]
    )

    hourly["hour_of_day"] = hourly["hour"].dt.hour
    hourly["day_of_week"] = hourly["hour"].dt.dayofweek
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)

    hourly["hour_sin"] = np.sin(
        2 * np.pi * hourly["hour_of_day"] / 24
    )
    hourly["hour_cos"] = np.cos(
        2 * np.pi * hourly["hour_of_day"] / 24
    )
    hourly["dow_sin"] = np.sin(
        2 * np.pi * hourly["day_of_week"] / 7
    )
    hourly["dow_cos"] = np.cos(
        2 * np.pi * hourly["day_of_week"] / 7
    )

    hourly = hourly.sort_values(
        ["station_id", "hour"]
    ).reset_index(drop=True)

    grouped = hourly.groupby("station_id", group_keys=False)

    for column in ["departures", "arrivals", "net_flow"]:
        hourly[f"{column}_lag_1h"] = grouped[column].shift(1)
        hourly[f"{column}_lag_2h"] = grouped[column].shift(2)
        hourly[f"{column}_lag_24h"] = grouped[column].shift(24)

        hourly[f"{column}_rolling_3h"] = grouped[column].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        hourly[f"{column}_rolling_24h"] = grouped[column].transform(
            lambda x: x.shift(1).rolling(24, min_periods=1).mean()
        )

    # Target de P2: desequilibrio de la siguiente hora.
    # El umbral binario Risk=1/0 se calculará en train.py
    # usando solamente el conjunto de entrenamiento para evitar leakage.
    hourly["target_departures"] = grouped["departures"].shift(-1)
    hourly["target_arrivals"] = grouped["arrivals"].shift(-1)
    hourly["target_net_flow"] = (
        hourly["target_departures"] - hourly["target_arrivals"]
    )
    hourly["target_imbalance_ratio"] = (
        hourly["target_net_flow"].abs() / hourly["capacity"]
    )

    hourly = hourly.dropna(
        subset=["target_imbalance_ratio"]
    ).copy()

    hourly = hourly.replace([np.inf, -np.inf], np.nan)

    return hourly


@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / "features_p2.parquet",
):
    """
    Genera el dataset de características para P2.

    Unidad: estación-hora.
    Objetivo: predecir el desequilibrio operacional de la siguiente hora.
    """
    logger.info("Generando features para P2...")

    trips = load_trips(PROCESSED_DATA_DIR)
    stations = load_stations(PROCESSED_DATA_DIR)

    features = build_station_hour_table(trips, stations)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    logger.success(
        f"Features P2 generadas: {output_path.name} "
        f"({len(features):,} observaciones)"
    )
    logger.info(
        "Risk=1/0 se calculará en train.py con el percentil 80 "
        "calculado únicamente sobre train."
    )


if __name__ == "__main__":
    app()
