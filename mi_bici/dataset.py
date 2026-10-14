import pandas as pd
import subprocess
from pathlib import Path

import typer
from loguru import logger

from mi_bici.config import (
    EXTERNAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR
)

app = typer.Typer()

def download_file(url: str, destination: Path) -> Path:
    """
    Descarga un archivo desde una URL mediante curl.

    Si el archivo ya existe y no está vacío, evita
    descargarlo nuevamente.

    Parameters
    ----------
    url : str
        URL de origen del archivo.
    destination : Path
        Ruta donde se guardará el archivo.

    Returns
    -------
    Path
        Ruta del archivo descargado.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists() and destination.stat().st_size > 0:
        logger.info(f"El archivo ya existe: {destination.name}")
        return destination

    temp_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    command = [
        "curl.exe",
        "--fail",
        "--location",
        "--retry", "2",
        "--connect-timeout", "15",
        "--output", str(temp_path),
        url
    ]

    try:
        logger.info(f"Descargando: {destination.name}")

        subprocess.run(
            command,
            check=True
        )

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError("El archivo descargado está vacío.")

        temp_path.replace(destination)

        logger.success(
            f"Descarga completada: {destination.name}"
        )

        return destination

    except (subprocess.CalledProcessError, OSError):
        temp_path.unlink(missing_ok=True)
        logger.exception(
            f"No fue posible descargar {destination.name}"
        )
        raise


def read_csv_robust(path: Path) -> tuple[pd.DataFrame, str]:
    """
    Lee un archivo CSV intentando distintas codificaciones.

    Parameters
    ----------
    path : Path
        Ruta del archivo CSV original.

    Returns
    -------
    tuple[pd.DataFrame, str]
        DataFrame y codificación utilizada.

    Raises
    ------
    UnicodeError
        Si ninguna codificación permite leer el archivo.
    """
    encodings = ["utf-8-sig", "cp1252"]

    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                dtype="string",
                low_memory=False,
            )

            logger.success(
                f"{path.name}: leído con {encoding}, "
                f"{len(df):,} registros"
            )

            return df, encoding

        except UnicodeError:
            logger.warning(
                f"{path.name}: no se pudo leer con {encoding}"
            )

    raise UnicodeError(
        f"No fue posible decodificar el archivo {path.name}"
    )


def normalize_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza los datos de viajes de MiBici.

    Conserva las variables operativas necesarias para
    P1, P2 y P3, convierte las fechas y calcula duración.

    No elimina registros: identifica posibles anomalías
    mediante indicadores de calidad.
    """
    column_mapping = {
        "Viaje_Id": "trip_id",
        "Inicio_del_viaje": "started_at",
        "Fin_del_viaje": "ended_at",
        "Origen_Id": "origin_station_id",
        "Destino_Id": "destination_station_id",
    }

    missing_columns = set(column_mapping) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing_columns)}"
        )

    trips = df[list(column_mapping)].rename(
        columns=column_mapping
    ).copy()

    # Normalizar identificadores
    for col in [
        "trip_id",
        "origin_station_id",
        "destination_station_id",
    ]:
        trips[col] = trips[col].astype("string").str.strip()

    # Convertir fechas
    trips["started_at"] = pd.to_datetime(
        trips["started_at"],
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
    )

    trips["ended_at"] = pd.to_datetime(
        trips["ended_at"],
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
    )

    # Calcular duración
    trips["duration_minutes"] = (
        trips["ended_at"] - trips["started_at"]
    ).dt.total_seconds() / 60

    # Indicadores de calidad
    trips["invalid_start"] = trips["started_at"].isna()
    trips["invalid_end"] = trips["ended_at"].isna()

    trips["negative_duration"] = (
        trips["duration_minutes"] < 0
    ).fillna(False)

    trips["zero_duration"] = (
        trips["duration_minutes"] == 0
    ).fillna(False)

    trips["duplicate_trip_id"] = trips.duplicated(
        subset=["trip_id"], keep=False
    )

    logger.success(
        f"Viajes normalizados: {len(trips):,} registros"
    )

    return trips


def normalize_stations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza la nomenclatura de estaciones de MiBici.

    Convierte coordenadas y capacidad a tipos numéricos,
    y señala las capacidades no válidas para el cálculo
    de indicadores operativos.
    """
    column_mapping = {
        "id": "station_id",
        "name": "station_name",
        "latitude": "latitude",
        "longitude": "longitude",
        "dpcapacity": "capacity",
    }

    missing_columns = set(column_mapping) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing_columns)}"
        )

    stations = df[list(column_mapping)].rename(
        columns=column_mapping
    ).copy()

    # Normalizar identificadores y nombres
    stations["station_id"] = (
        stations["station_id"].astype("string").str.strip()
    )

    stations["station_name"] = (
        stations["station_name"].astype("string").str.strip()
    )

    # Convertir variables numéricas
    stations["latitude"] = pd.to_numeric(
        stations["latitude"], errors="coerce"
    )

    stations["longitude"] = pd.to_numeric(
        stations["longitude"], errors="coerce"
    )

    stations["capacity"] = pd.to_numeric(
        stations["capacity"], errors="coerce"
    ).astype("Int64")

    # Reglas de calidad
    stations["capacity_valid"] = (
        stations["capacity"] > 0
    ).fillna(False)

    stations["coordinates_valid"] = (
        stations["latitude"].between(-90, 90)
        & stations["longitude"].between(-180, 180)
    ).fillna(False)

    stations["duplicate_station_id"] = stations.duplicated(
        subset=["station_id"], keep=False
    )

    logger.success(
        f"Estaciones normalizadas: {len(stations):,} registros"
    )

    return stations


def save_parquet(
    df: pd.DataFrame,
    destination: Path,
) -> Path:
    """
    Guarda un DataFrame en formato Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Datos previamente normalizados.
    destination : Path
        Ruta del archivo Parquet de salida.

    Returns
    -------
    Path
        Ruta donde se guardó el archivo.
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = destination.with_suffix(".parquet.part")

    try:
        df.to_parquet(
            temp_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        temp_path.replace(destination)

        logger.success(
            f"Parquet generado: {destination.name} "
            f"({len(df):,} registros)"
        )

        return destination

    except Exception:
        temp_path.unlink(missing_ok=True)
        logger.exception(
            f"Error al guardar {destination.name}"
        )
        raise

@app.command()
def main():
    # Rutas de los archivos originales
    trips_path = RAW_DATA_DIR / "datos_abiertos_2026_01.csv"
    stations_path = (
        EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv"
    )

    # Lectura robusta
    trips_raw, trips_encoding = read_csv_robust(trips_path)
    stations_raw, stations_encoding = read_csv_robust(stations_path)

    # Normalización
    trips = normalize_trips(trips_raw)
    stations = normalize_stations(stations_raw)

    # Guardar los datos procesados
    trips_output = (
        PROCESSED_DATA_DIR / "viajes_2026_01.parquet"
    )

    stations_output = (
        PROCESSED_DATA_DIR / "estaciones.parquet"
    )

    save_parquet(trips, trips_output)
    save_parquet(stations, stations_output)

    print("\n--- RESULTADOS ---")
    print(f"Viajes: {len(trips):,}")
    print(f"Estaciones: {len(stations):,}")
    print(f"Codificación viajes: {trips_encoding}")
    print(f"Codificación estaciones: {stations_encoding}")

    print("\n--- CALIDAD DE ESTACIONES ---")
    print(
        stations[
            [
                "capacity_valid",
                "coordinates_valid",
                "duplicate_station_id",
            ]
        ].value_counts(dropna=False)
    )

    print("\n--- CORRESPONDENCIA DE ESTACIONES ---")
    valid_ids = set(stations["station_id"].dropna())

    origins_missing = (
        ~trips["origin_station_id"].isin(valid_ids)
    ).sum()

    destinations_missing = (
        ~trips["destination_station_id"].isin(valid_ids)
    ).sum()

    print(f"Viajes con origen sin catálogo: {origins_missing:,}")
    print(f"Viajes con destino sin catálogo: {destinations_missing:,}")

if __name__ == "__main__":
    app()
