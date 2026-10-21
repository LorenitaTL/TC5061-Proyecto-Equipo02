"""Entrenamiento del primer modelo candidato para P1."""

from loguru import logger
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mi_bici.config import P1_SAMPLE_SIZE, RANDOM_STATE
from mi_bici.data_tracking import log_training_data
from mi_bici.features_p1 import DATASET_PATH, FEATURES
from mi_bici.modeling.evaluate_p1 import predict_demand, regression_metrics
from mi_bici.modeling.plots_p1 import log_validation_plots
from mi_bici.modeling.splits_p1 import temporal_masks
from mi_bici.modeling.tracking import configure_mlflow, log_run_config
from mi_bici.modeling.tracking_p1 import log_p1_traceability


def run(model: str, params: dict[str, str]) -> str:
    """Ejecuta un modelo de P1 y devuelve su identificador de corrida."""
    model_name = model.lower()

    if model_name == "seasonal":
        if params:
            raise ValueError("El baseline seasonal no recibe hiperparámetros.")

        from mi_bici.modeling.baseline_p1 import main as run_baseline

        return run_baseline()

    if model_name not in {"default", "histgb"}:
        raise ValueError(f"Modelo P1 desconocido: {model}. Usa histgb o seasonal.")

    converters = {
        "sample_size": int,
        "max_iter": int,
        "max_leaf_nodes": int,
        "learning_rate": float,
        "l2_regularization": float,
        "loss": str,
    }

    unknown = set(params) - set(converters)
    if unknown:
        raise ValueError(f"Parámetros P1 desconocidos: {sorted(unknown)}")

    effective = {}
    for key, value in params.items():
        try:
            effective[key] = converters[key](value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Valor inválido para {key}: {value}") from error

    sample_size = effective.pop("sample_size", P1_SAMPLE_SIZE)
    if sample_size < 1:
        raise ValueError("sample_size debe ser mayor que cero.")

    df = pd.read_parquet(
        DATASET_PATH,
        columns=["hour", "trip_count", *FEATURES],
    )

    masks = temporal_masks(df["hour"])

    train = df.loc[masks["train"]]
    validation = df.loc[masks["validation"]]

    train_sample = train.sample(
        n=min(sample_size, len(train)),
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

    model.set_params(**effective)

    configure_mlflow()

    with mlflow.start_run(run_name="hist_gradient_boosting_v2") as run:
        log_p1_traceability(
            model_family="histgb",
            stage_of_work="candidate",
        )

        log_training_data(
            X_train.assign(trip_count=y_train.to_numpy()),
            name="p1_training_sample",
            source=DATASET_PATH.resolve().as_uri(),
            target="trip_count",
        )

        log_run_config(
            {
                "model": "HistGradientBoostingRegressor",
                "hyperparameters": model.get_params(),
                "sample_size_requested": sample_size,
                "sample_size_actual": len(X_train),
                "n_validation": len(validation),
                "features": list(FEATURES),
                "target": "trip_count",
                "dataset": DATASET_PATH.name,
            }
        )

        logger.info("Iniciando entrenamiento...")
        model.fit(X_train, y_train)

        logger.info("Calculando predicciones...")
        predictions = predict_demand(model, validation)
        metrics = regression_metrics(
            validation["trip_count"], predictions, "val"
        )
        log_validation_plots(validation["trip_count"], predictions)

        mlflow.log_params(
            {
                "estimator_class": "HistGradientBoostingRegressor",
                "loss": model.loss,
                "max_iter": model.max_iter,
                "max_leaf_nodes": model.max_leaf_nodes,
                "learning_rate": model.learning_rate,
                "l2_regularization": model.l2_regularization,
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

        input_example = X_train.head(5).copy()
        signature = infer_signature(
            input_example,
            model.predict(input_example),
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="modelo_p1",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ],
        )

        logger.success(f"Run ID: {run.info.run_id}")
        logger.success(f"MAE validación: {metrics['mae_val']:.4f}")
        logger.success(f"RMSE validación: {metrics['rmse_val']:.4f}")

    logger.success("Entrenamiento finalizado.")

    return run.info.run_id

def main() -> None:
    """Conserva la ejecución directa del candidato con valores predeterminados."""
    run(model="histgb", params={})

if __name__ == "__main__":
    main()
