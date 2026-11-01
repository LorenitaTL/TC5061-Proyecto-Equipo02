"""Pruebas del baseline estacional de P1."""

import pandas as pd

from mi_bici.modeling.baseline_p1 import seasonal_prediction


def test_seasonal_prediction_uses_fallback_order():
    data = pd.DataFrame(
        {
            "lag_168h": [8.0, None, None, None, 0.0],
            "lag_24h": [9.0, 5.0, None, None, 7.0],
            "lag_1h": [10.0, 6.0, 3.0, None, 9.0],
        },
        index=[10, 20, 30, 40, 50],
    )
    original = data.copy(deep=True)

    result = seasonal_prediction(data)

    expected = pd.Series(
        [8.0, 5.0, 3.0, 0.0, 0.0],
        index=data.index,
        name="lag_168h",
    )
    pd.testing.assert_series_equal(result, expected)
    pd.testing.assert_frame_equal(data, original)


def test_seasonal_prediction_accepts_empty_data():
    data = pd.DataFrame(
        {column: pd.Series(dtype="float64") for column in ("lag_168h", "lag_24h", "lag_1h")}
    )

    result = seasonal_prediction(data)

    pd.testing.assert_series_equal(result, data["lag_168h"])
