"""Trazabilidad específica de los experimentos P2."""

import mlflow

from mi_bici.config import (
    DATA_END_MONTH,
    DATA_START_MONTH,
    P2_TRAIN_END,
    P2_VALIDATION_END,
    RANDOM_STATE,
    RISK_PERCENTILE,
)
from mi_bici.modeling.tracking import log_traceability


def log_p2_traceability(
    model_family: str | None = None,
    stage_of_work: str | None = None,
) -> None:
    """Registra la trazabilidad específica de la pregunta P2."""

    log_traceability(
        "P2",
        model_family=model_family,
        stage_of_work=stage_of_work,
    )

    mlflow.log_params(
        {
            "data_start_month": DATA_START_MONTH,
            "data_end_month": DATA_END_MONTH,
            "train_end": P2_TRAIN_END,
            "validation_end": P2_VALIDATION_END,
            "risk_percentile": RISK_PERCENTILE,
            "random_state": RANDOM_STATE,
            "unit_of_analysis": "station-hour",
        }
    )
