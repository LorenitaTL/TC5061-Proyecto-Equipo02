"""Registro del baseline estacional de demanda horaria en MLflow."""

import json

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mi_bici.config import PROCESSED_DATA_DIR
from mi_bici.modeling.splits_p1 import temporal_masks
from mi_bici.modeling.tracking import configure_mlflow
from mi_bici.modeling.tracking_p1 import log_p1_traceability

DATASET = PROCESSED_DATA_DIR / "p1_demanda_horaria.parquet"


def main() -> None:
    """Evalúa y registra un baseline estacional reproducible."""
    df = pd.read_parquet(
        DATASET,
        columns=[
            "hour",
            "trip_count",
            "lag_1h",
            "lag_24h",
            "lag_168h",
        ],
    )

    masks = temporal_masks(df["hour"])
    validation = df.loc[masks["validation"]].copy()

    if validation.empty:
        raise ValueError("La partición de validación está vacía.")

    y_true = validation["trip_count"]
    y_pred = (
        validation["lag_168h"].fillna(validation["lag_24h"]).fillna(validation["lag_1h"]).fillna(0)
    )

    metrics = {
        "mae_val": float(mean_absolute_error(y_true, y_pred)),
        "rmse_val": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "n_validation": len(validation),
        "zero_predictions_pct": float(100 * (y_pred == 0).mean()),
    }

    experiment_name = configure_mlflow()

    with mlflow.start_run(run_name="baseline_semanal_v1") as run:
        log_p1_traceability()
        mlflow.log_params(
            {
                "problem": "P1_regresion_demanda_horaria",
                "baseline": "lag_168h_lag_24h_lag_1h_zero",
                "target": "trip_count",
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.log_dict(
            {
                "description": (
                    "Baseline estacional: usa demanda de la misma "
                    "hora de la semana anterior. Si no existe, "
                    "utiliza lag de 24 horas, luego de 1 hora "
                    "y finalmente cero."
                ),
                "dataset": DATASET.name,
                "unit_of_analysis": "estacion-hora",
                "limitations": [
                    "Los viajes registrados representan demanda observada.",
                    "No existe historial de disponibilidad de estaciones.",
                    "Los ceros imputados no garantizan operación activa.",
                ],
            },
            "baseline_metadata.json",
        )

        logger.success(f"Run ID: {run.info.run_id}")
        logger.success(f"MAE validación: {metrics['mae_val']:.4f}")
        logger.success(f"RMSE validación: {metrics['rmse_val']:.4f}")
        logger.success(f"Experimento: {experiment_name}")

    print("\nMétricas registradas:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
