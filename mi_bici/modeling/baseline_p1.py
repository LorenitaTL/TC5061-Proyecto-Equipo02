"""Registro del baseline estacional de demanda horaria en MLflow."""

import json

from loguru import logger
import mlflow
import pandas as pd

from mi_bici.features_p1 import DATASET_PATH
from mi_bici.modeling.evaluate_p1 import regression_metrics
from mi_bici.modeling.splits_p1 import temporal_masks
from mi_bici.modeling.tracking import configure_mlflow
from mi_bici.modeling.tracking_p1 import log_p1_traceability


def seasonal_prediction(df: pd.DataFrame) -> pd.Series:
    """Usa lag_168h; si falta, lag_24h, luego lag_1h y finalmente cero."""
    return df["lag_168h"].fillna(df["lag_24h"]).fillna(df["lag_1h"]).fillna(0)


def main() -> str:
    """Evalúa y registra un baseline estacional reproducible."""
    df = pd.read_parquet(
        DATASET_PATH,
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
    test = df.loc[masks["test"]].copy()

    if validation.empty:
        raise ValueError("La partición de validación está vacía.")
    if test.empty:
        raise ValueError("La partición de prueba está vacía.")

    y_pred_val = seasonal_prediction(validation)
    y_pred_test = seasonal_prediction(test)

    metrics = {
        **regression_metrics(validation["trip_count"], y_pred_val, "val"),
        **regression_metrics(test["trip_count"], y_pred_test, "test"),
        "n_validation": len(validation),
        "n_test": len(test),
        "zero_predictions_pct_val": float(100 * (y_pred_val == 0).mean()),
        "zero_predictions_pct_test": float(100 * (y_pred_test == 0).mean()),
    }

    experiment_name = configure_mlflow()

    with mlflow.start_run(run_name="baseline_semanal_v1") as run:
        log_p1_traceability(
            model_family="seasonal",
            stage_of_work="baseline",
        )
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
                "dataset": DATASET_PATH.name,
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
        logger.success(f"MAE prueba: {metrics['mae_test']:.4f}")
        logger.success(f"RMSE prueba: {metrics['rmse_test']:.4f}")
        logger.success(f"Experimento: {experiment_name}")

    print("\nMétricas registradas:")
    print(json.dumps(metrics, indent=2))

    return run.info.run_id




if __name__ == "__main__":
    main()
