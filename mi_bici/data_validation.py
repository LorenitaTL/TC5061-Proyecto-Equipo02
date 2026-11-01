"""Validación de los datos procesados del proyecto MiBici."""

import json

import pandas as pd


def inspect_p1_structure(df: pd.DataFrame) -> dict[str, int]:
    """Resume problemas estructurales del dataset horario de P1."""
    required = ["station_id", "hour", "trip_count"]
    missing_columns = sorted(set(required) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {missing_columns}")

    if df.empty:
        raise ValueError("El dataset de P1 está vacío.")

    if not pd.api.types.is_datetime64_any_dtype(df["hour"]):
        raise ValueError("La columna hour debe tener un tipo datetime.")

    null_counts = df[required].isna().sum()

    # Solo las filas con clave completa permiten revisar la continuidad.
    ordered = df.dropna(subset=["station_id", "hour"]).sort_values(["station_id", "hour"])
    intervals = ordered.groupby("station_id")["hour"].diff()
    invalid_intervals = intervals.notna() & intervals.ne(pd.Timedelta(hours=1))

    return {
        "rows": len(df),
        "missing_station_id": int(null_counts["station_id"]),
        "missing_hour": int(null_counts["hour"]),
        "missing_trip_count": int(null_counts["trip_count"]),
        "duplicate_station_hours": int(ordered.duplicated(["station_id", "hour"]).sum()),
        "non_hourly_intervals": int(invalid_intervals.sum()),
    }


def inspect_p1_lags(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Revisa los nulos y valores históricos de los rezagos de P1."""
    structure = inspect_p1_structure(df)
    problems = {key: value for key, value in structure.items() if key != "rows" and value > 0}
    if problems:
        raise ValueError(f"Corrige la estructura antes de validar los rezagos: {problems}")

    lags = (1, 24, 168)
    required = [f"lag_{lag}h" for lag in lags]
    missing_columns = sorted(set(required) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Faltan columnas de rezagos: {missing_columns}")

    ordered = df.sort_values(["station_id", "hour"]).reset_index(drop=True)
    grouped = ordered.groupby("station_id")
    position = grouped.cumcount()
    results = {}

    for lag in lags:
        column = f"lag_{lag}h"
        actual = ordered[column]
        expected = grouped["trip_count"].shift(lag)

        initial = position < lag
        missing = actual.isna()
        comparable = actual.notna() & expected.notna()

        results[column] = {
            "expected_initial_nulls": int((missing & initial).sum()),
            "unexpected_nulls": int((missing & ~initial).sum()),
            "values_without_history": int((actual.notna() & initial).sum()),
            "incorrect_values": int((comparable & actual.ne(expected)).sum()),
        }

    return results


def p1_required_checks():
    """Construye los checks estrictos para los campos obligatorios de P1."""
    from deepchecks.tabular.checks import DataDuplicates, PercentOfNulls

    return [
        PercentOfNulls(
            columns=["station_id", "hour", "trip_count"],
        ).add_condition_percent_of_nulls_not_greater_than(0),
        DataDuplicates(
            columns=["station_id", "hour"],
        ).add_condition_ratio_less_or_equal(0),
    ]


def run_p1_required_validation(df: pd.DataFrame):
    """Valida los campos obligatorios sin tratarlos como datos de modelado."""
    from deepchecks.tabular import Dataset, Suite

    columns = ["station_id", "hour", "trip_count"]
    dataset = Dataset(
        df[columns],
        features=columns,
        cat_features=["station_id"],
    )
    suite = Suite("P1 required fields", *p1_required_checks())
    return suite.run(dataset)


def main() -> None:
    """Valida P1 y guarda el resumen estructural y el reporte DeepChecks."""
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity

    from mi_bici.config import RANDOM_STATE, REPORTS_DIR
    from mi_bici.features_p1 import DATASET_PATH, FEATURES

    df = pd.read_parquet(DATASET_PATH)

    structure = inspect_p1_structure(df)
    lags = inspect_p1_lags(df)

    sample = df.sample(
        n=min(10_000, len(df)),
        random_state=RANDOM_STATE,
    )
    dataset = Dataset(
        sample,
        label="trip_count",
        features=FEATURES,
        cat_features=["hour_of_day", "day_of_week", "is_weekend"],
    )
    result = data_integrity().run(dataset)
    required_result = run_p1_required_validation(sample)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": DATASET_PATH.name,
        "structure": structure,
        "lags": lags,
        "deepchecks_sample_rows": len(sample),
        "random_state": RANDOM_STATE,
    }
    summary_path = REPORTS_DIR / "p1_data_validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    html_path = REPORTS_DIR / "p1_data_integrity.html"
    with html_path.open("w", encoding="utf-8") as output:
        result.save_as_html(
            output,
            as_widget=False,
            connected=False,
        )

    required_path = REPORTS_DIR / "p1_required_fields.html"
    with required_path.open("w", encoding="utf-8") as output:
        required_result.save_as_html(
            output,
            as_widget=False,
            connected=False,
        )

    print(f"Reporte de campos obligatorios: {required_path}")

    print(f"Resumen: {summary_path}")
    print(f"Reporte DeepChecks: {html_path}")


if __name__ == "__main__":
    main()
