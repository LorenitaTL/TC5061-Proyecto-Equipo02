"""Pruebas de construcción y reproducibilidad del candidato P1."""

import numpy as np
from sklearn.metrics import mean_absolute_error

from mi_bici.config import RANDOM_STATE
from mi_bici.modeling.train_p1 import build_model


def test_build_model_configuration() -> None:
    """Conserva los valores del candidato y permite ajustar parámetros."""
    model = build_model()
    params = model.get_params()

    assert params["random_state"] == RANDOM_STATE
    assert params["loss"] == "squared_error"
    assert params["max_iter"] == 100
    assert params["max_leaf_nodes"] == 31
    assert params["learning_rate"] == 0.1
    assert params["l2_regularization"] == 1.0

    adjusted_model = build_model(max_iter=25, learning_rate=0.05)

    assert adjusted_model.max_iter == 25
    assert adjusted_model.learning_rate == 0.05
    assert model.max_iter == 100


def test_training_is_reproducible() -> None:
    """Dos ajustes independientes generan iguales predicciones y MAE."""
    rng = np.random.default_rng(RANDOM_STATE)

    X = rng.normal(size=(400, 4)).astype("float32")
    expected_count = np.exp(1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1])
    y = rng.poisson(expected_count)

    X_train, X_test = X[:300], X[300:]
    y_train, y_test = y[:300], y[300:]

    model_a = build_model()
    model_b = build_model()

    assert model_a is not model_b

    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)

    predictions_a = model_a.predict(X_test)
    predictions_b = model_b.predict(X_test)

    # Evita que la comparación se limite a predicciones constantes.
    assert np.ptp(predictions_a) > 0

    np.testing.assert_allclose(
        predictions_a,
        predictions_b,
        rtol=0,
        atol=1e-12,
    )

    mae_a = mean_absolute_error(y_test, predictions_a)
    mae_b = mean_absolute_error(y_test, predictions_b)

    np.testing.assert_allclose(mae_a, mae_b, rtol=0, atol=1e-12)
    