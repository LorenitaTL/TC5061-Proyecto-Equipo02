
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mi_bici.config import PROCESSED_DATA_DIR

app = typer.Typer()


def validate_station_coverage(
    trips_path: Path,
    catalog_ids: set[str],
) -> dict:
    """
    Evalúa la correspondencia de estaciones de un mes
    con el catálogo de referencia.
    """
    trips = pd.read_parquet(
        trips_path,
        columns=[
            "origin_station_id",
            "destination_station_id",
        ],
    )

    origin = trips["origin_station_id"].astype("string")
    destination = trips["destination_station_id"].astype("string")

    missing_origin = origin.isna() | origin.str.strip().eq("").fillna(False)
    missing_destination = (
        destination.isna()
        | destination.str.strip().eq("").fillna(False)
    )

    unknown_origin = (
        ~origin.isin(catalog_ids) & ~missing_origin
    )
    unknown_destination = (
        ~destination.isin(catalog_ids) & ~missing_destination
    )

    return {
        "period": trips_path.stem.replace("viajes_", "").replace("_", "-"),
        "rows": len(trips),
        "missing_origin": int(missing_origin.sum()),
        "missing_destination": int(missing_destination.sum()),
        "unknown_origins": int(unknown_origin.sum()),
        "unknown_destinations": int(unknown_destination.sum()),
    }


@app.command()
def main():
    """Genera un reporte de cobertura de estaciones."""
    stations = pd.read_parquet(
        PROCESSED_DATA_DIR / "estaciones.parquet",
        columns=["station_id"],
    )

    catalog_ids = set(
        stations["station_id"].dropna().astype("string")
    )

    files = sorted(
        PROCESSED_DATA_DIR.glob("viajes_????_??.parquet")
    )

    if not files:
        raise FileNotFoundError(
            "No se encontraron archivos de viajes procesados."
        )

    results = [
        validate_station_coverage(path, catalog_ids)
        for path in files
    ]

    # Resultados de correspondencia de estaciones
    station_summary = pd.DataFrame(results)

    # Indicadores de calidad generados durante la ingesta
    ingestion_path = (
        PROCESSED_DATA_DIR
        / "resumen_ingesta_2025-01_2026-06.csv"
    )

    if not ingestion_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el resumen de ingesta: {ingestion_path}"
        )

    ingestion_summary = pd.read_csv(ingestion_path)

    # Combinar las métricas utilizando el periodo
    summary = ingestion_summary.merge(
        station_summary.drop(columns=["rows"]),
        on="period",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not summary["_merge"].eq("both").all():
        raise ValueError(
            "Los periodos del resumen de ingesta "
            "y de cobertura de estaciones no coinciden."
        )

    summary = summary.drop(columns=["_merge"])

    # Guardar reporte consolidado
    output_path = (
        PROCESSED_DATA_DIR / "data_quality_report.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(summary.to_string(index=False))

    logger.success(
        f"Reporte de calidad generado: {output_path}"
    )


if __name__ == "__main__":
    app()
