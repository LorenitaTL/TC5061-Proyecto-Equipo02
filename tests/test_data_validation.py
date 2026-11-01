"""Pruebas de las reglas estructurales y de rezagos de P1."""

import pandas as pd
import pytest

from mi_bici.data_validation import inspect_p1_lags, inspect_p1_structure


@pytest.fixture
def clean_data():
    """Una estación con 180 horas continuas y rezagos conocidos."""
    df = pd.DataFrame(
        {
            "station_id": ["A"] * 180,
            "hour": pd.date_range("2026-01-01", periods=180, freq="h"),
            "trip_count": range(180),
        }
    )
    for lag in (1, 24, 168):
        df[f"lag_{lag}h"] = [float("nan")] * lag + list(range(180 - lag))
    return df


def test_clean_structure(clean_data):
    result = inspect_p1_structure(clean_data)

    assert result["rows"] == 180
    assert all(value == 0 for key, value in result.items() if key != "rows")


def test_clean_lags(clean_data):
    result = inspect_p1_lags(clean_data)

    for lag in (1, 24, 168):
        assert result[f"lag_{lag}h"] == {
            "expected_initial_nulls": lag,
            "unexpected_nulls": 0,
            "values_without_history": 0,
            "incorrect_values": 0,
        }


@pytest.mark.parametrize(
    "row,value,indicator",
    [
        (170, float("nan"), "unexpected_nulls"),
        (0, 10.0, "values_without_history"),
        (170, 999.0, "incorrect_values"),
    ],
)
def test_detects_lag_problems(clean_data, row, value, indicator):
    clean_data.loc[row, "lag_168h"] = value

    result = inspect_p1_lags(clean_data)

    assert result["lag_168h"][indicator] == 1


@pytest.mark.parametrize(
    "column,indicator",
    [
        ("station_id", "missing_station_id"),
        ("hour", "missing_hour"),
        ("trip_count", "missing_trip_count"),
    ],
)
def test_detects_missing_required_values(clean_data, column, indicator):
    # Permite introducir un faltante sin forzar tipos incompatibles.
    damaged = clean_data.astype({column: "object"})
    damaged.loc[10, column] = None
    if column == "hour":
        damaged["hour"] = pd.to_datetime(damaged["hour"])

    result = inspect_p1_structure(damaged)

    assert result[indicator] == 1


def test_detects_duplicate_station_hour(clean_data):
    damaged = pd.concat(
        [clean_data, clean_data.iloc[[10]]],
        ignore_index=True,
    )

    result = inspect_p1_structure(damaged)

    assert result["duplicate_station_hours"] == 1


def test_detects_hour_gap_and_blocks_lag_validation(clean_data):
    damaged = clean_data.drop(index=10)

    result = inspect_p1_structure(damaged)

    assert result["non_hourly_intervals"] == 1

    with pytest.raises(ValueError, match="Corrige la estructura"):
        inspect_p1_lags(damaged)


@pytest.mark.parametrize("problem", [None, "null", "duplicate"])
def test_deepchecks_required_conditions(clean_data, problem):
    pytest.importorskip("deepchecks")
    from mi_bici.data_validation import run_p1_required_validation

    data = clean_data[["station_id", "hour", "trip_count"]].copy()

    if problem == "null":
        data.loc[10, "hour"] = pd.NaT
    elif problem == "duplicate":
        data = pd.concat([data, data.iloc[[10]]], ignore_index=True)

    result = run_p1_required_validation(data)

    # Un error de ejecución no debe confundirse con detectar datos inválidos.
    assert not result.get_not_ran_checks()

    if problem is None:
        assert result.passed()
    else:
        assert not result.passed()


@pytest.mark.parametrize("introduce_drift", [False, True])
def test_detects_feature_and_label_drift(introduce_drift):
    pytest.importorskip("deepchecks")

    from mi_bici.data_validation import run_p1_drift_validation

    hours = pd.Series(range(240))
    train = pd.DataFrame(
        {
            "hour_of_day": hours % 24,
            "day_of_week": (hours // 24) % 7,
            "is_weekend": ((hours // 24) % 7 >= 5).astype(int),
            "lag_1h": hours % 20,
            "lag_24h": hours % 30,
            "lag_168h": hours % 40,
            "trip_count": hours % 25,
        }
    )
    test = train.copy()

    if introduce_drift:
        test["lag_1h"] += 1_000
        test["trip_count"] += 1_000

    result = run_p1_drift_validation(train, test)

    assert not result.get_not_ran_checks()
    feature_result, label_result = result.results

    feature_score = feature_result.value["lag_1h"]["Drift score"]
    label_score = label_result.value["Drift score"]

    if introduce_drift:
        assert feature_score > 0.9
        assert label_score > 0.9
        assert not feature_result.passed_conditions()
    else:
        assert feature_score == pytest.approx(0)
        assert label_score == pytest.approx(0)
        assert feature_result.passed_conditions()


def test_drift_samples_respect_time_split_and_are_reproducible():
    from mi_bici.data_validation import prepare_p1_drift_samples
    from mi_bici.features_p1 import FEATURES
    from mi_bici.modeling.splits_p1 import TRAIN_END, VALIDATION_END

    # Cinco filas por partición, incluyendo las fechas de corte.
    hours = pd.DatetimeIndex(
        [
            *pd.date_range(end=TRAIN_END - pd.Timedelta(hours=1), periods=5, freq="h"),
            *pd.date_range(start=TRAIN_END, periods=5, freq="h"),
            *pd.date_range(start=VALIDATION_END, periods=5, freq="h"),
        ]
    )
    data = pd.DataFrame(
        {
            "station_id": ["A"] * len(hours),
            "hour": hours,
            "trip_count": range(len(hours)),
        }
    )
    for column in FEATURES:
        data[column] = 0

    original = data.copy(deep=True)

    train, test = prepare_p1_drift_samples(data, sample_size=3)
    repeated_train, repeated_test = prepare_p1_drift_samples(data, sample_size=3)

    assert len(train) == len(test) == 3
    assert (train["hour"] < TRAIN_END).all()
    assert (test["hour"] >= VALIDATION_END).all()
    pd.testing.assert_frame_equal(train, repeated_train)
    pd.testing.assert_frame_equal(test, repeated_test)
    pd.testing.assert_frame_equal(data, original)

    # Si hay menos filas que las solicitadas, conserva todas sin duplicarlas.
    full_train, full_test = prepare_p1_drift_samples(data, sample_size=100)
    assert len(full_train) == len(full_test) == 5
    assert full_train["hour"].is_unique
    assert full_test["hour"].is_unique
