import numpy as np
import pytest
from sklearn.datasets import make_classification

from mi_bici.modeling.train_p2 import build_pipeline, get_classifier


@pytest.mark.parametrize(
    "model_family",
    [
        "LogisticRegression",
        "HistGradientBoosting",
        "RandomForest",
    ],
)
def test_p2_training_is_reproducible(model_family: str) -> None:
    """Dos entrenamientos con los mismos datos y configuración producen las mismas probabilidades."""

    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        random_state=123,
    )

    model_1 = build_pipeline(get_classifier(model_family))
    model_2 = build_pipeline(get_classifier(model_family))

    model_1.fit(X, y)
    model_2.fit(X, y)

    probabilities_1 = model_1.predict_proba(X)[:, 1]
    probabilities_2 = model_2.predict_proba(X)[:, 1]

    predictions_1 = model_1.predict(X)
    predictions_2 = model_2.predict(X)

    np.testing.assert_allclose(
        probabilities_1,
        probabilities_2,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        predictions_1,
        predictions_2,
    )
