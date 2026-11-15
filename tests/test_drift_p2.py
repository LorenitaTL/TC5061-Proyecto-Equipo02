"""Pruebas del reporte de drift y calidad de P2 (``mi_bici.monitoring.drift_p2``).

No usan ``data/``: construyen conjuntos sintéticos con las 30 variables del modelo.
"""

import json

import numpy as np
import pandas as pd
import pytest

from mi_bici.modeling.train_p2 import ablation_features
from mi_bici.monitoring.drift_p2 import (
    CATEGORICAL_FEATURES,
    LABEL,
    load_reference,
    main,
    sample_rows,
    to_model_frame,
)
from mi_bici.monitoring.prepare_p2 import RISK_THRESHOLD, TARGET

FEATURES = ablation_features("completo")


def make_features(n_rows: int, seed: int, start: str = "2025-01-01", shift: float = 0.0):
    """Tabla estación-hora sintética con las columnas y tipos de features_p2.parquet."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, periods=n_rows, freq="h")
    data = {
        "station_id": pd.array([f"S{i % 5}" for i in range(n_rows)], dtype="string"),
        "hour": hours,
    }
    for column in FEATURES:
        data[column] = rng.poisson(1.0, n_rows).astype("float64") + shift
    data["hour_of_day"] = hours.hour.astype("int32")
    data["day_of_week"] = hours.dayofweek.astype("int32")
    data["is_weekend"] = (hours.dayofweek >= 5).astype("int8")
    data["capacity"] = pd.array(rng.integers(10, 30, n_rows), dtype="Int64")
    data["imbalance_ratio"] = pd.array(rng.uniform(0, 0.2, n_rows), dtype="Float64")
    data[TARGET] = pd.array(rng.uniform(0, 0.2, n_rows), dtype="Float64")
    return pd.DataFrame(data)


def as_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Imita la salida de prepare_p2: clave, variables y Risk."""
    risk = (df[TARGET] > RISK_THRESHOLD).astype("int8")
    return df[["station_id", "hour", *FEATURES]].assign(**{LABEL: risk})


def test_load_reference_keeps_training_period_and_labels_with_train_threshold(tmp_path):
    path = tmp_path / "features_p2.parquet"
    # 2026-03-31 00:00 + 48 h cruza el corte de entrenamiento (2026-04-01, exclusivo).
    data = make_features(48, seed=1, start="2026-03-31")
    data.loc[3, TARGET] = pd.NA
    data.to_parquet(path, index=False)

    reference = load_reference(path)

    assert reference["hour"].max() < pd.Timestamp("2026-04-01")
    assert len(reference) == 23  # 24 horas de marzo menos la fila sin objetivo
    expected = (data.set_index("hour").loc[reference["hour"], TARGET] > RISK_THRESHOLD).to_numpy()
    assert (reference[LABEL].to_numpy() == expected.astype("int8")).all()
    assert list(reference.columns) == ["station_id", "hour", *FEATURES, LABEL]


def test_sample_rows_is_reproducible_and_bounded():
    data = as_evaluation(make_features(200, seed=2))

    first = sample_rows(data, 50, seed=42)
    second = sample_rows(data, 50, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 50
    assert len(sample_rows(data, 1_000)) == len(data)
    with pytest.raises(ValueError):
        sample_rows(data, 0)


def test_to_model_frame_converts_nullable_types_for_evidently():
    data = as_evaluation(make_features(30, seed=3))
    data.loc[0, "capacity"] = pd.NA

    frame = to_model_frame(data, FEATURES)

    assert list(frame.columns) == [*FEATURES, LABEL]
    assert frame["capacity"].dtype == "float64"
    assert np.isnan(frame.loc[0, "capacity"])
    for column in CATEGORICAL_FEATURES:
        assert frame[column].dtype == "int64"
    assert frame[LABEL].dtype == "int64"


def test_to_model_frame_rejects_missing_columns():
    data = as_evaluation(make_features(10, seed=4)).drop(columns=["capacity"])

    with pytest.raises(ValueError, match="capacity"):
        to_model_frame(data, FEATURES)


@pytest.mark.parametrize("shift", [0.0, 5.0])
def test_report_detects_drift_only_when_distribution_changes(tmp_path, shift):
    pytest.importorskip("evidently")
    features_path = tmp_path / "features_p2.parquet"
    evaluation_path = tmp_path / "evaluation.parquet"
    output_dir = tmp_path / "reports"
    make_features(600, seed=5).to_parquet(features_path, index=False)
    # Mismos valores que la referencia, con o sin desplazamiento: así el resultado no
    # depende del ruido de muestreo (con 500 filas ya ronda el umbral de 0.1).
    current = make_features(600, seed=5, shift=shift)
    as_evaluation(current).to_parquet(evaluation_path, index=False)

    main(
        features_path=features_path,
        evaluation_path=evaluation_path,
        output_dir=output_dir,
        sample_size=500,
        seed=42,
    )

    for name in (
        "p2_drift_calidad.html",
        "p2_drift_por_variable.csv",
        "p2_drift_resumen.json",
        "p2_drift_por_variable.png",
        "p2_distribuciones_drift.png",
    ):
        assert (output_dir / name).stat().st_size > 0
    table = pd.read_csv(output_dir / "p2_drift_por_variable.csv")
    summary = json.loads((output_dir / "p2_drift_resumen.json").read_text(encoding="utf-8"))
    assert set(table["variable"]) == {*FEATURES, LABEL}

    if shift:
        # Las numéricas desplazadas (no las categóricas de calendario) deben marcar drift.
        assert table.set_index("variable").loc["departures", "drift"]
        assert summary["share_predictores_con_drift"] > 0.5
    else:
        assert summary["predictores_con_drift"] == []
