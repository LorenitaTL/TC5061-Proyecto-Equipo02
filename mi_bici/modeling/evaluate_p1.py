"""Evaluación final del modelo P1 sobre junio de 2026."""

import argparse

from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mi_bici.features_p1 import DATASET_PATH, FEATURES
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


def predict_demand(model, df: pd.DataFrame) -> np.ndarray:
    """Predice los viajes de cada estación-hora; la demanda no puede ser negativa."""
    return np.clip(model.predict(df[FEATURES].astype("float32")), 0, None)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray, split: str) -> dict[str, float]:
    """Calcula MAE y RMSE con el conjunto en el nombre, por ejemplo ``mae_val``."""
    return {
        f"mae_{split}": float(mean_absolute_error(y_true, y_pred)),
        f"rmse_{split}": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main() -> None:
    """Evalúa el modelo previamente seleccionado sin reentrenarlo."""
    args = parse_args()
    configure_mlflow()

    logger.info("Cargando modelo registrado...")
    model = mlflow.sklearn.load_model(args.model_uri)

    df = pd.read_parquet(
        DATASET_PATH,
        columns=["hour", "trip_count", *FEATURES],
    )

    masks = temporal_masks(df["hour"])
    test = df.loc[masks["test"]].copy()

    if test.empty:
        raise ValueError("La partición de prueba está vacía.")

    logger.info("Generando predicciones de prueba...")
    predictions = predict_demand(model, test)
    metrics = {
        **regression_metrics(test["trip_count"], predictions, "test"),
        "n_test": len(test),
    }

    # Viajes reales y predichos por día en todo el sistema, para revisar el error en el tiempo.
    daily = (
        test.assign(predicted=predictions)
        .groupby(test["hour"].dt.date)[["trip_count", "predicted"]]
        .sum()
        .rename_axis("date")
        .rename(columns={"trip_count": "real"})
        .round(1)
    )

    with mlflow.start_run(run_name="evaluacion_final_p1_v2"):
        log_p1_traceability()
        mlflow.log_param("model_uri", args.model_uri)
        mlflow.log_param("test_period", "2026-06")
        mlflow.log_metrics(metrics)
        mlflow.log_text(daily.to_csv(), "test_daily_trips.csv")
        mlflow.log_dict(
            {
                "description": "Evaluación final del modelo registrado, sin reentrenarlo.",
                "model_uri": args.model_uri,
                "dataset": DATASET_PATH.name,
                "unit_of_analysis": "estacion-hora",
            },
            "evaluation_metadata.json",
        )

    logger.success(f"MAE prueba: {metrics['mae_test']:.4f}")
    logger.success(f"RMSE prueba: {metrics['rmse_test']:.4f}")
    logger.success(f"Observaciones: {metrics['n_test']:,}")


if __name__ == "__main__":
    main()
