"""Funciones de feature engineering comunes a P1, P2 y P3.

Cada pregunta construye sus variables en su propio archivo, con el mismo sufijo que su
entrenamiento en ``mi_bici/modeling``, y reutiliza las funciones de este módulo:

- ``features_p1``: dataset estación-hora con variables calendario y rezagos para predecir
  los viajes iniciados en la siguiente hora (P1, regresión).
- ``features_p2``: flujos, desbalance relativo a la capacidad, variables temporales,
  rezagos y ventanas móviles para predecir el riesgo de la siguiente hora (P2,
  clasificación).
- ``features_p3``: perfiles de uso por estación, con el balance de flujo en horas pico y
  el uso en fin de semana (P3, clustering).

Las funciones de este módulo solo usan información disponible hasta el momento de cada
observación: cargan datos, cuentan viajes en la hora de su timestamp y miran al pasado.
"""

from pathlib import Path

from loguru import logger
import pandas as pd

from mi_bici.config import DATA_END_MONTH, DATA_START_MONTH, PROCESSED_DATA_DIR


def month_range(start_month: str, end_month: str) -> list[str]:
    """Devuelve los meses del rango, incluidos ambos extremos, en el formato ``AAAA_MM`` de los
    archivos de viajes."""
    return list(pd.period_range(start_month, end_month, freq="M").strftime("%Y_%m"))


def trip_files(
    start_month: str = DATA_START_MONTH,
    end_month: str = DATA_END_MONTH,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> list[Path]:
    """Devuelve las rutas de los Parquet mensuales de viajes del rango.

    Falla si falta algún mes, para no construir variables con un periodo incompleto sin
    notarlo.
    """
    paths = [
        processed_dir / f"viajes_{month}.parquet" for month in month_range(start_month, end_month)
    ]

    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"No se encontraron los archivos de viajes: {missing}")

    return paths


def load_trips(
    columns: list[str],
    start_month: str = DATA_START_MONTH,
    end_month: str = DATA_END_MONTH,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Carga las columnas pedidas de los viajes normalizados del rango de meses.

    Leer solo las columnas que usa cada pregunta reduce la memoria necesaria: el periodo
    completo tiene más de seis millones de viajes.
    """
    trips = pd.concat(
        [
            pd.read_parquet(path, columns=columns)
            for path in trip_files(start_month, end_month, processed_dir)
        ],
        ignore_index=True,
    )

    logger.success(f"Viajes cargados ({start_month} a {end_month}): {len(trips):,}")

    return trips


def load_stations(
    columns: list[str],
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Carga el catálogo de estaciones con ``station_id`` y las columnas pedidas.

    ``station_id`` se normaliza a texto sin espacios para que coincida con los
    identificadores de origen y destino de los viajes, y se conserva una fila por estación.
    La capacidad y la ubicación son atributos de la estación, no del comportamiento futuro de
    los viajes.
    """
    path = processed_dir / "estaciones.parquet"

    if not path.is_file():
        raise FileNotFoundError(f"No se encontró {path}")

    stations = pd.read_parquet(path, columns=["station_id", *columns])
    stations["station_id"] = stations["station_id"].astype("string").str.strip()

    return stations.drop_duplicates("station_id", keep="last").reset_index(drop=True)


def count_station_hourly(
    trips: pd.DataFrame,
    station_column: str,
    time_column: str,
    name: str,
) -> pd.DataFrame:
    """Cuenta los viajes por estación y hora, con columnas ``station_id``, ``hour`` y ``name``.

    Con el origen y la hora de inicio se obtienen las salidas; con el destino y la hora de
    fin, las llegadas. Cada viaje se asigna a la hora de su timestamp observado, así que el
    conteo de una hora no usa viajes posteriores.
    """
    return (
        trips.assign(hour=trips[time_column].dt.floor("h"))
        .groupby([station_column, "hour"], as_index=False)
        .size()
        .rename(columns={station_column: "station_id", "size": name})
    )


def add_calendar_features(df: pd.DataFrame, time_column: str = "hour") -> pd.DataFrame:
    """Agrega ``hour_of_day``, ``day_of_week`` y ``is_weekend`` a partir de ``time_column``.

    Los viajes y el desbalance de una estación cambian con la hora del día y entre días
    laborales y fin de semana. Las variables se obtienen solo del timestamp de cada
    observación, sin información futura. Modifica ``df`` y lo devuelve.
    """
    df["hour_of_day"] = df[time_column].dt.hour
    df["day_of_week"] = df[time_column].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    return df


def add_station_lags(
    df: pd.DataFrame,
    column: str,
    lags: list[int],
    prefix: str = "",
) -> pd.DataFrame:
    """Agrega rezagos de ``column`` por estación: ``{prefix}lag_{n}h`` es su valor ``n`` horas
    antes.

    El comportamiento reciente y el de la misma hora de días o semanas anteriores son los
    mejores indicadores de la siguiente hora. Supone una fila por estación y hora sin huecos,
    por lo que cada pregunta completa la cuadrícula estación-hora antes de llamarla. Ordena
    por estación y hora para que ``shift(n)`` corresponda a ``n`` horas antes y solo mire al
    pasado. Las primeras horas de cada estación quedan vacías por falta de historia.
    """
    df = df.sort_values(["station_id", "hour"]).reset_index(drop=True)
    grouped = df.groupby("station_id")[column]

    for lag in lags:
        df[f"{prefix}lag_{lag}h"] = grouped.shift(lag)

    return df
