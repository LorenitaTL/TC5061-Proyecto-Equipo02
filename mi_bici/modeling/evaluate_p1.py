
"""Evaluación final del modelo P1 sobre junio de 2026."""

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mi_bici.config import PROCESSED_DATA_DIR

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/m-2068e241c1ef41f4b735648a1fd89f0d"

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
    mlflow.set_tracking_uri(TRACKING_URI)

    logger.info("Cargando modelo registrado...")
    model = mlflow.sklearn.load_model(MODEL_URI)

    dataset = PROCESSED_DATA_DIR / "p1_demanda_horaria.parquet"

    df = pd.read_parquet(
        dataset,
        columns=["hour", "trip_count", *FEATURES],
    )

    test = df.loc[df["hour"] >= "2026-06-01"].copy()

    X_test = test[FEATURES].astype("float32")
    y_test = test["trip_count"]

    logger.info("Generando predicciones de prueba...")
    predictions = np.clip(model.predict(X_test), 0, None)

    metrics = {
        "mae_test": float(
            mean_absolute_error(y_test, predictions)
        ),
        "rmse_test": float(
            np.sqrt(mean_squared_error(y_test, predictions))
        ),
        "n_test": len(test),
    }

    with mlflow.start_run(run_name="evaluacion_final_p1_v2"):
        mlflow.log_param("model_uri", MODEL_URI)
        mlflow.log_param("test_period", "2026-06")
        mlflow.log_metrics(metrics)

    logger.success(f"MAE prueba: {metrics['mae_test']:.4f}")
    logger.success(f"RMSE prueba: {metrics['rmse_test']:.4f}")
    logger.success(f"Observaciones: {metrics['n_test']:,}")


if __name__ == "__main__":
    main()
