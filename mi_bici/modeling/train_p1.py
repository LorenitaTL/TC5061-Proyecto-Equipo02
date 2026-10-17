"""Entrenamiento del primer modelo candidato para P1."""

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mi_bici.config import PROCESSED_DATA_DIR
from mi_bici.modeling.splits_p1 import temporal_masks
from mi_bici.modeling.tracking_p1 import configure_mlflow, log_p1_traceability

DATASET = PROCESSED_DATA_DIR / "p1_demanda_horaria.parquet"

FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
]

RANDOM_STATE = 42
SAMPLE_SIZE = 1_000_000


def main() -> None:
    """Entrena, evalúa y registra el modelo candidato."""

    df = pd.read_parquet(
        DATASET,
        columns=["hour", "trip_count", *FEATURES],
    )

    masks = temporal_masks(df["hour"])

    train = df.loc[masks["train"]]
    validation = df.loc[masks["validation"]]

    train_sample = train.sample(
        n=min(SAMPLE_SIZE, len(train)),
        random_state=RANDOM_STATE,
    )

    X_train = train_sample[FEATURES].astype("float32")
    y_train = train_sample["trip_count"]

    X_val = validation[FEATURES].astype("float32")
    y_val = validation["trip_count"]

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=100,
        max_leaf_nodes=31,
        learning_rate=0.1,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    configure_mlflow()

    with mlflow.start_run(run_name="hist_gradient_boosting_v2") as run:
        log_p1_traceability()
        logger.info("Iniciando entrenamiento...")
        model.fit(X_train, y_train)

        logger.info("Calculando predicciones...")
        predictions = model.predict(X_val)

        # La demanda no puede ser negativa.
        predictions = np.clip(predictions, 0, None)

        metrics = {
            "mae_val": float(mean_absolute_error(y_val, predictions)),
            "rmse_val": float(np.sqrt(mean_squared_error(y_val, predictions))),
        }

        mlflow.log_params(
            {
                "model": "HistGradientBoostingRegressor",
                "loss": "squared_error",
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "learning_rate": 0.1,
                "l2_regularization": 1.0,
                "random_state": RANDOM_STATE,
                "sample_size": len(train_sample),
                "features": ",".join(FEATURES),
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.log_dict(
            {
                "target": "trip_count",
                "unit": "station-hour",
                "dataset": DATASET.name,
                "training": "Muestra aleatoria reproducible del periodo de entrenamiento",
                "evaluation": "Validación cronológica abril-mayo 2026",
                "limitations": [
                    "Los conteos reflejan viajes realizados, no demanda insatisfecha.",
                    "La disponibilidad histórica de estaciones es desconocida.",
                    "No se utilizan identificadores de estación como predictores.",
                ],
            },
            "model_metadata.json",
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo_p1",
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ],
        )

        logger.success(f"Run ID: {run.info.run_id}")
        logger.success(f"MAE validación: {metrics['mae_val']:.4f}")
        logger.success(f"RMSE validación: {metrics['rmse_val']:.4f}")

    logger.success("Entrenamiento finalizado.")


if __name__ == "__main__":
    main()
