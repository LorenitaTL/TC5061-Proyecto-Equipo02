"""Trazabilidad específica de los experimentos P2."""

import mlflow

from mi_bici.config import (
    DATA_END_MONTH,
    DATA_START_MONTH,
    RANDOM_STATE,
    RISK_PERCENTILE,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from mi_bici.modeling.tracking import log_traceability


def log_p2_traceability() -> None:
    """Registra la trazabilidad específica de la pregunta P2."""

    log_traceability("P2")

    mlflow.log_params(
        {
            "data_start_month": DATA_START_MONTH,
            "data_end_month": DATA_END_MONTH,
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": TEST_FRACTION,
            "risk_percentile": RISK_PERCENTILE,
            "random_state": RANDOM_STATE,
            "unit_of_analysis": "station-hour",
        }
    )
