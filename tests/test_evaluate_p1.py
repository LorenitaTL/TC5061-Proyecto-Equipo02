"""Pruebas de métricas y predicciones de demanda de P1."""

import numpy as np
import pandas as pd
import pytest

from mi_bici.features_p1 import FEATURES
from mi_bici.modeling.evaluate_p1 import predict_demand, regression_metrics


@pytest.mark.parametrize("split", ["val", "test"])
def test_regression_metrics_known_errors(split):
    # Errores absolutos: 0 y 4; MAE = 2; MSE = 8.
    result = regression_metrics(
        pd.Series([0.0, 4.0]),
        np.array([0.0, 0.0]),
        split,
    )

    assert result == {
        f"mae_{split}": pytest.approx(2.0),
        f"rmse_{split}": pytest.approx(np.sqrt(8.0)),
    }


def test_regression_metrics_single_observation():
    result = regression_metrics(
        pd.Series([3.0]),
        np.array([5.0]),
        "test",
    )

    assert result == {"mae_test": 2.0, "rmse_test": 2.0}


def test_regression_metrics_perfect_predictions():
    values = pd.Series([0.0, 2.0, 5.0])

    result = regression_metrics(values, values.to_numpy(), "test")

    assert result == {"mae_test": 0.0, "rmse_test": 0.0}


def test_regression_metrics_rejects_empty_inputs():
    with pytest.raises(ValueError):
        regression_metrics(
            pd.Series(dtype="float64"),
            np.array([], dtype=float),
            "test",
        )


def test_predict_demand_prepares_features_and_clips_negatives():
    class ToyModel:
        def predict(self, features):
            self.received = features.copy()
            return np.array([-2.0, 0.0, 3.5])

    # Orden distinto y una columna extra para verificar la entrada al modelo.
    data = pd.DataFrame(
        {column: [1, 2, 3] for column in reversed(FEATURES)},
        index=[10, 20, 30],
    )
    data["station_id"] = "A"
    data.loc[20, "lag_168h"] = np.nan
    original = data.copy(deep=True)
    model = ToyModel()

    result = predict_demand(model, data)

    np.testing.assert_array_equal(result, [0.0, 0.0, 3.5])
    pd.testing.assert_frame_equal(
        model.received,
        data[FEATURES].astype("float32"),
    )
    pd.testing.assert_frame_equal(data, original)
