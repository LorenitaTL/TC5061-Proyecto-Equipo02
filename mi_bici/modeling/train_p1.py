"""Entrenamiento del primer modelo candidato para P1."""

from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mi_bici.config import P1_SAMPLE_SIZE, RANDOM_STATE
from mi_bici.features_p1 import DATASET_PATH, FEATURES
from mi_bici.modeling.evaluate_p1 import predict_demand, regression_metrics
from mi_bici.modeling.splits_p1 import temporal_masks
from mi_bici.modeling.tracking import configure_mlflow
from mi_bici.modeling.tracking_p1 import log_p1_traceability

SAMPLE_SIZE = P1_SAMPLE_SIZE


def main() -> None:
    """Entrena, evalúa y registra el modelo candidato."""

    df = pd.read_parquet(
        DATASET_PATH,
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
        metrics = regression_metrics(
            validation["trip_count"], predict_demand(model, validation), "val"
        )

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
                "dataset": DATASET_PATH.name,
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
