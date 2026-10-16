from datetime import UTC, datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urljoin

from loguru import logger
import pandas as pd
import typer

from mi_bici.config import (
    DATA_END_MONTH,
    DATA_START_MONTH,
    EXTERNAL_DATA_DIR,
    MIBICI_BASE_URL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REFERENCES_DIR,
    STATIONS_FILENAME,
)

app = typer.Typer()

PORTAL_URL = "https://mibici.net/es/datos-abiertos/"


def get_curl_executable() -> str:
    """Localiza curl según el sistema operativo."""

    executable = "curl.exe" if sys.platform == "win32" else "curl"

    resolved = shutil.which(executable)

    if resolved is None:
        raise RuntimeError(f"No se encontró {executable}. Instala curl para ejecutar la ingesta.")

    return resolved


class CSVLinkParser(HTMLParser):
    """Extrae enlaces CSV de la página de datos abiertos."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return

        attributes = dict(attrs)
        href = attributes.get("href", "")

        if ".csv" in href.lower():
            self.links.append(urljoin(PORTAL_URL, href))


def discover_sources(
    start_month: str,
    end_month: str,
) -> dict[str, str]:
    """
    Descubre los archivos mensuales de MiBici.

    Parameters
    ----------
    start_month : str
        Mes inicial en formato YYYY-MM.
    end_month : str
        Mes final en formato YYYY-MM.

    Returns
    -------
    dict[str, str]
        Diccionario con periodo y URL de descarga.
    """
    response = subprocess.run(
        [
            get_curl_executable(),
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--max-time",
            "60",
            PORTAL_URL,
        ],
        capture_output=True,
        check=True,
    )

    html = response.stdout.decode("utf-8-sig")

    parser = CSVLinkParser()
    parser.feed(html)

    sources = {}

    for url in parser.links:
        match = re.search(
            r"datos_abiertos_(\d{4})_(\d{2})\.csv",
            url,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        year, month = map(int, match.groups())

        if not 1 <= month <= 12:
            continue

        period = f"{year:04d}-{month:02d}"

        if start_month <= period <= end_month:
            if period in sources and sources[period] != url:
                raise ValueError(f"Se encontraron URL diferentes para {period}")
            sources[period] = url

    expected = (
        pd.period_range(
            start=start_month,
            end=end_month,
            freq="M",
        )
        .strftime("%Y-%m")
        .tolist()
    )

    missing = sorted(set(expected) - set(sources))

    if missing:
        raise RuntimeError(f"No se encontraron los meses: {missing}")

    logger.success(f"Fuentes identificadas: {len(sources)} meses")

    return dict(sorted(sources.items()))


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

    temp_path = destination.with_suffix(destination.suffix + ".part")

    command = [
        get_curl_executable(),
        "--fail",
        "--location",
        "--retry",
        "2",
        "--connect-timeout",
        "15",
        "--output",
        str(temp_path),
        url,
    ]

    try:
        logger.info(f"Descargando: {destination.name}")

        subprocess.run(command, check=True)

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError("El archivo descargado está vacío.")

        temp_path.replace(destination)

        logger.success(f"Descarga completada: {destination.name}")

        return destination

    except (subprocess.CalledProcessError, OSError):
        temp_path.unlink(missing_ok=True)
        logger.exception(f"No fue posible descargar {destination.name}")
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

            logger.success(f"{path.name}: leído con {encoding}, {len(df):,} registros")

            return df, encoding

        except UnicodeError:
            logger.warning(f"{path.name}: no se pudo leer con {encoding}")

    raise UnicodeError(f"No fue posible decodificar el archivo {path.name}")


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
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")

    trips = df[list(column_mapping)].rename(columns=column_mapping).copy()

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
    trips["duration_minutes"] = (trips["ended_at"] - trips["started_at"]).dt.total_seconds() / 60

    # Indicadores de calidad
    trips["invalid_start"] = trips["started_at"].isna()
    trips["invalid_end"] = trips["ended_at"].isna()

    trips["negative_duration"] = (trips["duration_minutes"] < 0).fillna(False)

    trips["zero_duration"] = (trips["duration_minutes"] == 0).fillna(False)

    trips["duplicate_trip_id"] = trips.duplicated(subset=["trip_id"], keep=False)

    logger.success(f"Viajes normalizados: {len(trips):,} registros")

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
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")

    stations = df[list(column_mapping)].rename(columns=column_mapping).copy()

    # Normalizar identificadores y nombres
    stations["station_id"] = stations["station_id"].astype("string").str.strip()

    stations["station_name"] = stations["station_name"].astype("string").str.strip()

    # Convertir variables numéricas
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")

    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")

    stations["capacity"] = pd.to_numeric(stations["capacity"], errors="coerce").astype("Int64")

    # Reglas de calidad
    stations["capacity_valid"] = (stations["capacity"] > 0).fillna(False)

    stations["coordinates_valid"] = (
        stations["latitude"].between(-90, 90) & stations["longitude"].between(-180, 180)
    ).fillna(False)

    stations["duplicate_station_id"] = stations.duplicated(subset=["station_id"], keep=False)

    logger.success(f"Estaciones normalizadas: {len(stations):,} registros")

    return stations


def process_stations() -> dict:
    """
    Descarga y normaliza el catálogo de estaciones.
    """
    catalog_url = f"{MIBICI_BASE_URL}/site/assets/files/1118/{STATIONS_FILENAME}"

    raw_path = EXTERNAL_DATA_DIR / STATIONS_FILENAME
    processed_path = PROCESSED_DATA_DIR / "estaciones.parquet"

    download_file(catalog_url, raw_path)

    raw_df, encoding = read_csv_robust(raw_path)
    stations = normalize_stations(raw_df)

    save_parquet(stations, processed_path)

    result = {
        "source_url": catalog_url,
        "encoding": encoding,
        "raw_rows": len(raw_df),
        "processed_rows": len(stations),
        "invalid_capacity": int((~stations["capacity_valid"]).sum()),
        "invalid_coordinates": int((~stations["coordinates_valid"]).sum()),
    }

    logger.success(f"Catálogo procesado: {len(stations):,} estaciones")

    return result


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

        logger.success(f"Parquet generado: {destination.name} ({len(df):,} registros)")

        return destination

    except Exception:
        temp_path.unlink(missing_ok=True)
        logger.exception(f"Error al guardar {destination.name}")
        raise


def process_month(period: str, url: str) -> dict:
    """
    Descarga, normaliza y guarda los viajes de un mes.

    Procesa cada archivo de manera independiente para
    evitar mantener varios meses en memoria.

    Parameters
    ----------
    period : str
        Mes de los viajes en formato YYYY-MM.
    url : str
        URL oficial del archivo CSV.

    Returns
    -------
    dict
        Metadatos y resultados de calidad del periodo.
    """
    year, month = period.split("_") if "_" in period else period.split("-")

    filename = f"datos_abiertos_{year}_{month}.csv"
    raw_path = RAW_DATA_DIR / filename

    output_path = PROCESSED_DATA_DIR / f"viajes_{year}_{month}.parquet"

    logger.info(f"Iniciando procesamiento del mes {period}")

    # 1. Descargar archivo original
    download_file(url, raw_path)

    # 2. Leer con codificación compatible
    raw_df, encoding = read_csv_robust(raw_path)

    # 3. Normalizar columnas y tipos
    trips = normalize_trips(raw_df)

    # 4. Validar cobertura temporal
    period_observed = trips["started_at"].dt.strftime("%Y-%m")

    outside_period = (period_observed.notna() & (period_observed != period)).sum()

    # 5. Exportar a Parquet
    save_parquet(trips, output_path)

    # 6. Resumir métricas de calidad
    result = {
        "period": period,
        "source_url": url,
        "encoding": encoding,
        "raw_rows": len(raw_df),
        "processed_rows": len(trips),
        "invalid_start": int(trips["invalid_start"].sum()),
        "invalid_end": int(trips["invalid_end"].sum()),
        "negative_duration": int(trips["negative_duration"].sum()),
        "zero_duration": int(trips["zero_duration"].sum()),
        "duplicate_trip_id": int(trips["duplicate_trip_id"].sum()),
        "outside_period": int(outside_period),
        "output_file": output_path.name,
    }

    logger.success(f"Mes {period} procesado: {len(trips):,} registros")

    # Liberar referencias antes del siguiente mes
    del raw_df, trips

    return result


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calcula el hash SHA-256 de un archivo.

    Lee el archivo en bloques para evitar cargarlo
    completamente en memoria.
    """
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def generate_manifest(
    summary_path: Path,
    manifest_path: Path,
    stations_filename: str,
) -> dict:
    """
    Construye el manifiesto de datos a partir
    de archivos ya procesados.
    """
    summary = pd.read_csv(summary_path)

    records = []

    for _, row in summary.iterrows():
        period = str(row["period"])
        year, month = period.split("-")

        raw_filename = f"datos_abiertos_{year}_{month}.csv"

        raw_path = RAW_DATA_DIR / raw_filename
        processed_path = PROCESSED_DATA_DIR / f"viajes_{year}_{month}.parquet"

        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)

        if not processed_path.is_file():
            raise FileNotFoundError(processed_path)

        records.append(
            {
                "period": period,
                "source_url": str(row["source_url"]),
                "encoding": str(row["encoding"]),
                "rows": int(row["processed_rows"]),
                "raw_file": raw_filename,
                "raw_bytes": raw_path.stat().st_size,
                "raw_sha256": calculate_sha256(raw_path),
                "processed_file": processed_path.name,
                "processed_bytes": (processed_path.stat().st_size),
                "processed_sha256": calculate_sha256(processed_path),
            }
        )

        logger.info(f"Hashes calculados: {period}")

    stations_raw = EXTERNAL_DATA_DIR / stations_filename
    stations_processed = PROCESSED_DATA_DIR / "estaciones.parquet"

    for path in [stations_raw, stations_processed]:
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = {
        "dataset": "MiBici Guadalajara",
        "source": "Portal de datos abiertos MiBici",
        "period_start": str(summary["period"].min()),
        "period_end": str(summary["period"].max()),
        "generated_at_utc": (datetime.now(UTC).isoformat()),
        "total_months": len(records),
        "total_trips": int(summary["processed_rows"].sum()),
        "monthly_files": records,
        "stations": {
            "source_file": stations_raw.name,
            "source_sha256": calculate_sha256(stations_raw),
            "processed_file": stations_processed.name,
            "processed_sha256": calculate_sha256(stations_processed),
            "catalog_period": "2026-09",
            "limitation": (
                "La capacidad registrada en el catálogo "
                "no necesariamente representa la capacidad "
                "histórica durante 2025-01 a 2026-06."
            ),
        },
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Escritura temporal para no dejar JSON incompleto
    temp_path = manifest_path.with_suffix(".json.part")

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temp_path.replace(manifest_path)

    logger.success(f"Manifiesto generado: {manifest_path}")

    return manifest


def verify_manifest(manifest_path: Path) -> bool:
    """Verifica integridad de archivos contra hashes registrados."""
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    errors = []

    def verify_file(path: Path, expected_hash: str):
        if not path.is_file():
            errors.append(f"Archivo faltante: {path}")
            return

        actual_hash = calculate_sha256(path)

        if actual_hash != expected_hash:
            errors.append(f"Hash diferente: {path}")

    for record in manifest["monthly_files"]:
        raw_path = RAW_DATA_DIR / record["raw_file"]
        processed_path = PROCESSED_DATA_DIR / record["processed_file"]

        verify_file(raw_path, record["raw_sha256"])
        verify_file(
            processed_path,
            record["processed_sha256"],
        )

    stations = manifest["stations"]

    verify_file(
        EXTERNAL_DATA_DIR / stations["source_file"],
        stations["source_sha256"],
    )

    verify_file(
        PROCESSED_DATA_DIR / stations["processed_file"],
        stations["processed_sha256"],
    )

    if errors:
        for error in errors:
            logger.error(error)

        raise RuntimeError(f"Falló la verificación de {len(errors)} archivo(s).")

    logger.success("Integridad verificada: todos los archivos coinciden con el manifiesto.")

    return True


@app.command()
def ingest(
    start_month: str = typer.Option(
        DATA_START_MONTH,
        help="Mes inicial en formato YYYY-MM",
    ),
    end_month: str = typer.Option(
        DATA_END_MONTH,
        help="Mes final en formato YYYY-MM",
    ),
):
    """
    Ejecuta la ingesta de viajes de MiBici para
    un rango mensual configurable.
    """
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", start_month):
        raise typer.BadParameter("start_month debe tener formato YYYY-MM")

    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", end_month):
        raise typer.BadParameter("end_month debe tener formato YYYY-MM")

    if start_month > end_month:
        raise typer.BadParameter("El mes inicial no puede ser posterior al final")

    logger.info("Preparando catálogo de estaciones")
    process_stations()

    sources = discover_sources(start_month, end_month)

    results = []

    for period, url in sources.items():
        result = process_month(period, url)
        results.append(result)

    summary = pd.DataFrame(results)

    summary_path = PROCESSED_DATA_DIR / f"resumen_ingesta_{start_month}_{end_month}.csv"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.success(f"Resumen de ingesta guardado: {summary_path.name}")

    print("\n--- RESUMEN DE INGESTA ---")
    print(summary.to_string(index=False))

    logger.success(f"Ingesta completada para {len(results)} meses")


@app.command("manifest")
def manifest_command(
    start_month: str = typer.Option(DATA_START_MONTH),
    end_month: str = typer.Option(DATA_END_MONTH),
):
    """Genera el manifiesto de los datos ya procesados."""
    summary_path = PROCESSED_DATA_DIR / f"resumen_ingesta_{start_month}_{end_month}.csv"

    manifest_path = REFERENCES_DIR / "data_manifest.json"

    generate_manifest(
        summary_path=summary_path,
        manifest_path=manifest_path,
        stations_filename=STATIONS_FILENAME,
    )


@app.command("verify")
def verify_command():
    """Verifica archivos contra los hashes registrados."""
    manifest_path = REFERENCES_DIR / "data_manifest.json"

    verify_manifest(manifest_path)


if __name__ == "__main__":
    app()
