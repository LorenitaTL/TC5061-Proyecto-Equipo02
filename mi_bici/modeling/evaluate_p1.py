"""Evaluación final del modelo P1 sobre junio de 2026."""

import argparse

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mi_bici.config import PROCESSED_DATA_DIR
from mi_bici.modeling.splits_p1 import temporal_masks
from mi_bici.modeling.tracking import configure_mlflow
from mi_bici.modeling.tracking_p1 import log_p1_traceability


def parse_args() -> argparse.Namespace:
    """Obtiene la URI del modelo que queremos evaluar."""
    parser = argparse.ArgumentParser(description="Evaluación final de demanda horaria MiBici P1")
    parser.add_argument(
        "--model-uri",
        required=True,
        help="URI del modelo registrado en MLflow",
    )
    return parser.parse_args()


FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
]


def main() -> None:
    """Evalúa el modelo previamente seleccionado sin reentrenarlo."""
    args = parse_args()
    configure_mlflow()

    logger.info("Cargando modelo registrado...")
    model = mlflow.sklearn.load_model(args.model_uri)

    dataset = PROCESSED_DATA_DIR / "p1_demanda_horaria.parquet"

    df = pd.read_parquet(
        dataset,
        columns=["hour", "trip_count", *FEATURES],
    )

    masks = temporal_masks(df["hour"])
    test = df.loc[masks["test"]].copy()

    X_test = test[FEATURES].astype("float32")
    y_test = test["trip_count"]

    logger.info("Generando predicciones de prueba...")
    predictions = np.clip(model.predict(X_test), 0, None)

    metrics = {
        "mae_test": float(mean_absolute_error(y_test, predictions)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "n_test": len(test),
    }

    with mlflow.start_run(run_name="evaluacion_final_p1_v2"):
        log_p1_traceability()
        mlflow.log_param("model_uri", args.model_uri)
        mlflow.log_param("test_period", "2026-06")
        mlflow.log_metrics(metrics)

    logger.success(f"MAE prueba: {metrics['mae_test']:.4f}")
    logger.success(f"RMSE prueba: {metrics['rmse_test']:.4f}")
    logger.success(f"Observaciones: {metrics['n_test']:,}")


if __name__ == "__main__":
    main()
