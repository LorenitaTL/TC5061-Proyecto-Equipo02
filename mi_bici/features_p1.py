"""Construcción del dataset analítico para P1: demanda horaria."""

from loguru import logger
import pandas as pd

from mi_bici.config import PROCESSED_DATA_DIR


def aggregate_hourly_trips() -> pd.DataFrame:
    """Agrega los viajes iniciados por estación y hora."""

    files = sorted(PROCESSED_DATA_DIR.glob("viajes_????_??.parquet"))

    if not files:
        raise FileNotFoundError("No se encontraron archivos de viajes procesados.")

    monthly = []

    for path in files:
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

        valid["hour"] = valid["started_at"].dt.floor("h")

        summary = (
            valid.groupby(["origin_station_id", "hour"])
            .size()
            .reset_index(name="trip_count")
            .rename(columns={"origin_station_id": "station_id"})
        )

        monthly.append(summary)
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
    """Incorpora variables calendario y rezagos históricos."""

    result = df.copy()

    result["hour_of_day"] = result["hour"].dt.hour
    result["day_of_week"] = result["hour"].dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int8")

    grouped = result.groupby("station_id")["trip_count"]

    for lag in [1, 24, 168]:
        result[f"lag_{lag}h"] = grouped.shift(lag)

    return result


def main() -> None:
    """Genera y almacena el dataset analítico de P1."""

    hourly = aggregate_hourly_trips()

    logger.info(f"Viajes agregados: {hourly['trip_count'].sum():,}")

    grid = build_hourly_grid(hourly)
    features = add_features(grid)

    output = PROCESSED_DATA_DIR / "p1_demanda_horaria.parquet"

    features.to_parquet(
        output,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    logger.success(f"Dataset P1 generado: {len(features):,} filas")
    logger.success(f"Archivo: {output}")


if __name__ == "__main__":
    main()
