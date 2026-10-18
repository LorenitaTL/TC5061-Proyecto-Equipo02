import numpy as np

from mi_bici.modeling.train_p3 import N_CLUSTERS, stability_ari


def test_stability_ari_is_one_for_well_separated_groups():
    """Con grupos claramente separados, el resultado no depende de la semilla."""
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]])[:N_CLUSTERS]
    X = np.vstack([center + rng.normal(scale=0.1, size=(20, 2)) for center in centers])

    assert stability_ari(X) == 1
