"""Pruebas de las ablaciones del modelo P2."""

from mi_bici.modeling.train_p2 import (
    ABLATIONS,
    FEATURE_COLUMNS,
    ablation_features,
    ablation_run_name,
)


def test_ablation_feature_counts() -> None:
    """Cada ablación conserva el número de variables reportado en el README."""
    assert len(ablation_features("completo")) == 30
    assert len(ablation_features("sin_desbalance_actual")) == 28
    assert len(ablation_features("sin_flujo_actual")) == 25


def test_ablations_only_remove_existing_features() -> None:
    """Las variables quitadas existen en el modelo completo."""
    for removed in ABLATIONS.values():
        assert set(removed) <= set(FEATURE_COLUMNS)


def test_ablation_run_names() -> None:
    """El modelo completo conserva su nombre de run; las ablaciones lo indican."""
    assert ablation_run_name("completo") == "P2-logistic-regression-baseline"
    assert ablation_run_name("sin_flujo_actual") == "P2-logistic-regression-sin-flujo-actual"
