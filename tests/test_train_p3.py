import numpy as np
import pytest

from mi_bici.modeling.train_p3 import (
    LOCATION_FEATURES,
    N_CLUSTERS,
    USAGE_FEATURES,
    resolve_config,
    run_name,
    stability_ari,
)


def test_stability_ari_is_one_for_well_separated_groups():
    """Con grupos claramente separados, el resultado no depende de la semilla."""
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]])[:N_CLUSTERS]
    X = np.vstack([center + rng.normal(scale=0.1, size=(20, 2)) for center in centers])

    assert stability_ari(X) == 1


def test_resolve_config_uses_profile_model_by_default():
    config = resolve_config("default", {})

    assert config["model"] == "kmeans_perfil"
    assert config["features"] == USAGE_FEATURES
    assert config["stage_of_work"] == "candidate"
    assert config["n_clusters"] == N_CLUSTERS


def test_resolve_config_marks_location_model_as_baseline():
    config = resolve_config("KMeans_Ubicacion", {})

    assert config["features"] == LOCATION_FEATURES
    assert config["stage_of_work"] == "baseline"


def test_resolve_config_converts_params():
    config = resolve_config(
        "kmeans_perfil",
        {"n_clusters": "4", "n_init": "20", "features": "balance_am, balance_dia"},
    )

    assert config["n_clusters"] == 4
    assert config["n_init"] == 20
    assert config["features"] == ["balance_am", "balance_dia"]


@pytest.mark.parametrize(
    "model, params",
    [
        ("dbscan", {}),
        ("kmeans_perfil", {"eps": "0.5"}),
        ("kmeans_perfil", {"n_clusters": "cuatro"}),
        ("kmeans_perfil", {"n_clusters": "1"}),
        ("kmeans_perfil", {"features": "balance_am,usuario_id"}),
    ],
)
def test_resolve_config_rejects_invalid_input(model, params):
    with pytest.raises(ValueError):
        resolve_config(model, params)


def test_run_name_describes_model_and_custom_features():
    assert run_name(resolve_config("kmeans_perfil", {"n_clusters": "4"})) == (
        "p3-kmeans-perfil-k4"
    )
    assert run_name(resolve_config("kmeans_perfil", {"features": "balance_am,balance_pm"})) == (
        "p3-kmeans-perfil-k3-balance_am+balance_pm"
    )
