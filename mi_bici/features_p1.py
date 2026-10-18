"""Construcción del dataset analítico para P1: demanda horaria."""

from loguru import logger
import pandas as pd

from mi_bici.config import PROCESSED_DATA_DIR
from mi_bici.features import (
    add_calendar_features,
    add_station_lags,
    count_station_hourly,
    trip_files,
)

DATASET_PATH = PROCESSED_DATA_DIR / "p1_demanda_horaria.parquet"

# Variables predictoras del modelo de P1, en el orden en que las recibe.
FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
]


def aggregate_hourly_trips() -> pd.DataFrame:
    """Agrega los viajes iniciados por estación y hora.

    Procesa un mes a la vez para limitar la memoria y suma los conteos al final.
    """
    monthly = []

    for path in trip_files():
        df = pd.read_parquet(
            path,
            columns=[
                "started_at",
                "origin_station_id",
                "invalid_start",
            ],
        )

        valid = df.loc[
            df["started_at"].notna() & df["origin_station_id"].notna() & ~df["invalid_start"]
        ].copy()

        monthly.append(
            count_station_hourly(valid, "origin_station_id", "started_at", "trip_count")
        )
        logger.info(f"Procesado: {path.name}")

    hourly = (
        pd.concat(monthly, ignore_index=True)
        .groupby(["station_id", "hour"], as_index=False)["trip_count"]
        .sum()
    )

    return hourly.sort_values(["station_id", "hour"]).reset_index(drop=True)


def build_hourly_grid(hourly: pd.DataFrame) -> pd.DataFrame:
    """Completa las horas desde la primera actividad de cada estación."""

    end_hour = hourly["hour"].max()
    stations = []

    for station_id, group in hourly.groupby("station_id"):
        first_hour = group["hour"].min()

        hours = pd.date_range(
            start=first_hour,
            end=end_hour,
            freq="h",
        )

        station_grid = pd.DataFrame(
            {
                "station_id": station_id,
                "hour": hours,
            }
        )

        stations.append(station_grid)

    grid = pd.concat(stations, ignore_index=True)

    result = grid.merge(
        hourly,
        on=["station_id", "hour"],
        how="left",
        validate="one_to_one",
    )

    result["trip_count"] = result["trip_count"].fillna(0).astype("int32")

    return result.sort_values(["station_id", "hour"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Incorpora variables calendario y rezagos de 1 hora, 1 día y 1 semana de ``trip_count``."""

    result = add_calendar_features(df.copy())

    return add_station_lags(result, "trip_count", [1, 24, 168])


def main() -> None:
    """Genera y almacena el dataset analítico de P1."""

    hourly = aggregate_hourly_trips()

    logger.info(f"Viajes agregados: {hourly['trip_count'].sum():,}")

    grid = build_hourly_grid(hourly)
    features = add_features(grid)

    features.to_parquet(
        DATASET_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    logger.success(f"Dataset P1 generado: {len(features):,} filas")
    logger.success(f"Archivo: {DATASET_PATH}")


if __name__ == "__main__":
    main()
