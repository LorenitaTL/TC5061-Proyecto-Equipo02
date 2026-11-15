"""Genera el conjunto común de evaluación A/B de P2 para junio de 2026."""

import argparse
from pathlib import Path

import pandas as pd

from mi_bici.modeling.train_p2 import ablation_features, create_risk_label

RISK_THRESHOLD = 0.06666666666666667
KEY = ["station_id", "hour"]
TARGET = "target_imbalance_ratio"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/monitoring_p2/evaluation.parquet"),
    )
    args = parser.parse_args()

    features = ablation_features("completo")
    june = pd.read_parquet(
        args.input,
        columns=[*KEY, *features, TARGET],
        engine="pyarrow",
        filters=[
            ("hour", ">=", pd.Timestamp("2026-06-01")),
            ("hour", "<", pd.Timestamp("2026-07-01")),
        ],
    )

    if june.empty:
        raise ValueError("El archivo no contiene observaciones de junio de 2026.")
    if june[KEY].isna().any().any() or june.duplicated(KEY).any():
        raise ValueError("Las claves estacion-hora deben ser completas y unicas.")

    eligible = june.dropna(subset=[TARGET]).sort_values(KEY)
    if eligible.empty:
        raise ValueError("Junio no contiene observaciones con objetivo disponible.")

    risk = create_risk_label(eligible, RISK_THRESHOLD).astype("int8")
    evaluation = eligible[[*KEY, *features]].assign(Risk=risk)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_parquet(args.output, index=False)

    print(f"Archivo generado: {args.output}")
    print(f"Filas de junio: {len(june):,}")
    print(f"Filas elegibles para A/B: {len(evaluation):,}")
    print(f"Variables de entrada: {len(features)}")
    print(f"Casos Risk = 1: {int(risk.sum()):,} ({risk.mean():.2%})")


if __name__ == "__main__":
    main()
