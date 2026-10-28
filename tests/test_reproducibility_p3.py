import numpy as np
from sklearn.metrics import adjusted_rand_score

from mi_bici.config import RANDOM_STATE
from mi_bici.modeling.train_p3 import USAGE_FEATURES, fit_kmeans


def noisy_profiles(seed: int = 0) -> np.ndarray:
    """Perfiles sin grupos claros: con ruido puro, K-Means depende de la inicialización."""
    return np.random.default_rng(seed).normal(size=(300, len(USAGE_FEATURES)))


def test_fit_kmeans_is_reproducible_with_same_seed():
    X = noisy_profiles()

    first = fit_kmeans(X, seed=RANDOM_STATE)
    second = fit_kmeans(X, seed=RANDOM_STATE)

    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_array_equal(first.cluster_centers_, second.cluster_centers_)


def test_fit_kmeans_uses_random_state_by_default():
    X = noisy_profiles()

    np.testing.assert_array_equal(fit_kmeans(X).labels_, fit_kmeans(X, seed=RANDOM_STATE).labels_)


def test_fit_kmeans_changes_with_another_seed_on_noisy_data():
    """Caso de control: si otra semilla diera el mismo resultado, la prueba anterior no
    demostraría que la semilla fija es la que hace reproducible el entrenamiento."""
    X = noisy_profiles()

    reference = fit_kmeans(X, seed=RANDOM_STATE)
    other = fit_kmeans(X, seed=RANDOM_STATE + 1)

    assert adjusted_rand_score(reference.labels_, other.labels_) < 1
