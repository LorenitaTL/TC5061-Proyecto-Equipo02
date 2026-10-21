"""Trazabilidad específica de los experimentos P1."""

import mlflow

from mi_bici.config import DATA_END_MONTH, DATA_START_MONTH
from mi_bici.modeling.splits_p1 import TRAIN_END, VALIDATION_END
from mi_bici.modeling.tracking import log_traceability


def log_p1_traceability(
    model_family: str | None = None,
    stage_of_work: str | None = None,
) -> None:
    """Registra el contrato común y las particiones temporales de P1."""
    log_traceability(
        "P1",
        model_family=model_family,
        stage_of_work=stage_of_work,
    )

    mlflow.log_params(
        {
            "data_start_month": DATA_START_MONTH,
            "data_end_month": DATA_END_MONTH,
            "train_start": f"{DATA_START_MONTH}-01",
            "train_end_exclusive": TRAIN_END.strftime("%Y-%m-%d"),
            "validation_start": TRAIN_END.strftime("%Y-%m-%d"),
            "validation_end_exclusive": VALIDATION_END.strftime("%Y-%m-%d"),
            "test_start": VALIDATION_END.strftime("%Y-%m-%d"),
        }
    )
