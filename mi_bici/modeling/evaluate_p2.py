"""Evaluación final del modelo P2 sobre el conjunto de prueba."""

import argparse
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from mlflow import MlflowClient

from mi_bici.config import PROJ_ROOT, RISK_PERCENTILE
from mi_bici.features_p2 import DATASET_PATH
from mi_bici.modeling.tracking import configure_mlflow
from mi_bici.modeling.tracking_p2 import log_p2_traceability
from mi_bici.modeling.train_p2 import (
    ablation_features,
    create_risk_label,
    evaluate_model,
    load_features,
    temporal_split,
)


MODEL_NAME = "mibici-equipo02-p2-riesgo"


def parse_args() -> argparse.Namespace:
    """Obtiene la URI del modelo que queremos evaluar."""
    parser = argparse.ArgumentParser(
        description="Evaluación final del modelo P2 de riesgo operacional MiBici"
    )
    parser.add_argument(
        "--model-uri",
        required=True,
        help="URI del modelo registrado en MLflow, por ejemplo "
        "models:/mibici-equipo02-p2-riesgo@champion",
    )
    return parser.parse_args()


def get_source_run_info(model_uri: str) -> dict:
    """
    Obtiene del Model Registry la información del run que originó el modelo.

    Esto permite recuperar la ablación utilizada por el modelo seleccionado
    sin tener que escribirla manualmente en el evaluador.
    """
    client = MlflowClient()

    if "@champion" in model_uri:
        model_name = model_uri.split("models:/", 1)[1].split("@", 1)[0]

        model_version = client.get_model_version_by_alias(
            model_name,
            "champion",
        )

    elif model_uri.startswith("models:/"):
        model_name = model_uri.split("models:/", 1)[1].split("/", 1)[0]
        version = model_uri.rsplit("/", 1)[-1]

        model_version = client.get_model_version(
            name=model_name,
            version=version,
        )

    else:
        raise ValueError(
            "Para la evaluación P2 se espera una URI de Model Registry, "
            "por ejemplo models:/mibici-equipo02-p2-riesgo@champion"
        )

    source_run_id = model_version.run_id

    source_run = client.get_run(source_run_id)

    return {
        "model_name": model_version.name,
        "model_version": model_version.version,
        "source_run_id": source_run_id,
        "source_run_name": source_run.info.run_name,
        "ablation": source_run.data.params.get("ablation"),
        "risk_percentile": float(
            source_run.data.params.get(
                "risk_percentile",
                RISK_PERCENTILE,
            )
        ),
        "train_risk_threshold": source_run.data.params.get(
            "train_risk_threshold"
        ),
        "features_data_version": source_run.data.params.get(
            "features_data_version"
        ),
    }


def main() -> None:
    """Evalúa el modelo registrado sobre TEST sin reentrenarlo."""
    args = parse_args()

    configure_mlflow()

    # ---------------------------------------------------------
    # 1. Obtener información del modelo registrado
    # ---------------------------------------------------------

    logger.info("Obteniendo información del modelo registrado...")

    model_info = get_source_run_info(args.model_uri)

    logger.info(
        f"Modelo: {model_info['model_name']} "
        f"v{model_info['model_version']}"
    )

    logger.info(
        f"Run de origen: {model_info['source_run_id']}"
    )

    ablation = model_info["ablation"]

    if ablation is None:
        raise ValueError(
            "No se encontró el parámetro 'ablation' en el run de origen."
        )

    logger.info(f"Ablación seleccionada: {ablation}")

    # ---------------------------------------------------------
    # 2. Cargar modelo
    # ---------------------------------------------------------

    logger.info("Cargando modelo registrado...")

    model = mlflow.sklearn.load_model(args.model_uri)

    logger.success("Modelo cargado correctamente.")

    # ---------------------------------------------------------
    # 3. Cargar datos
    # ---------------------------------------------------------

    logger.info(f"Cargando dataset P2: {DATASET_PATH}")

    df = load_features(DATASET_PATH)

    # ---------------------------------------------------------
    # 4. División temporal
    # ---------------------------------------------------------

    train, validation, test = temporal_split(df)

    if train.empty:
        raise ValueError("La partición TRAIN está vacía.")

    if test.empty:
        raise ValueError("La partición TEST está vacía.")

    logger.info(
        f"Observaciones TEST: {len(test):,}"
    )

    # ---------------------------------------------------------
    # 5. Reconstruir el threshold usando SOLO TRAIN
    # ---------------------------------------------------------

    risk_percentile = model_info["risk_percentile"]

    risk_threshold = train["target_imbalance_ratio"].quantile(
        risk_percentile
    )

    logger.info(
        f"Umbral de riesgo P{risk_percentile * 100:.0f} "
        f"calculado sobre TRAIN: {risk_threshold:.6f}"
    )

    # ---------------------------------------------------------
    # 6. Crear etiqueta Risk para TEST
    # ---------------------------------------------------------

    test = test.copy()

    test["Risk"] = create_risk_label(
        test,
        risk_threshold,
    )

    # ---------------------------------------------------------
    # 7. Seleccionar exactamente las variables del modelo
    # ---------------------------------------------------------

    features = ablation_features(ablation)

    logger.info(
        f"Ablación: {ablation} | "
        f"Variables utilizadas: {len(features)}"
    )

    X_test = test[features]
    y_test = test["Risk"]

    # ---------------------------------------------------------
    # 8. Evaluación
    # ---------------------------------------------------------

    logger.info("Generando predicciones sobre TEST...")

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "test",
    )

    # ---------------------------------------------------------
    # 9. Información adicional de TEST
    # ---------------------------------------------------------

    metrics = {
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "n_test": float(len(test)),
        "test_risk_rate": float(y_test.mean()),
        "train_risk_threshold": float(risk_threshold),
    }

    # ---------------------------------------------------------
    # 10. Registrar evaluación final en MLflow
    # ---------------------------------------------------------

    with mlflow.start_run(
        run_name="evaluacion-final-p2"
    ) as run:

        log_p2_traceability()

        mlflow.set_tags(
            {
                "project": "MiBici",
                "question": "P2",
                "model_family": "LogisticRegression",
                "stage_of_work": "evaluation",
                "evaluation_type": "final_test",
                "model_uri": args.model_uri,
                "model_name": model_info["model_name"],
                "model_version": str(model_info["model_version"]),
                "source_run_id": model_info["source_run_id"],
                "source_run_name": model_info["source_run_name"],
                "ablation": ablation,
            }
        )

        mlflow.log_params(
            {
                "model_uri": args.model_uri,
                "source_run_id": model_info["source_run_id"],
                "model_version": model_info["model_version"],
                "ablation": ablation,
                "risk_percentile": risk_percentile,
                "features": ",".join(features),
                "n_features": len(features),
                "dataset": DATASET_PATH.name,
                "evaluation_split": "test",
                "evaluation_method": "registered_model_without_retraining",
            }
        )

        mlflow.log_metrics(metrics)

        # Periodo utilizado para la evaluación
        mlflow.log_dict(
            {
                "description": (
                    "Evaluación final del modelo P2 registrado "
                    "sobre el conjunto TEST, sin reentrenamiento."
                ),
                "model_uri": args.model_uri,
                "model_name": model_info["model_name"],
                "model_version": model_info["model_version"],
                "source_run_id": model_info["source_run_id"],
                "source_run_name": model_info["source_run_name"],
                "ablation": ablation,
                "risk_percentile": risk_percentile,
                "risk_threshold": float(risk_threshold),
                "dataset": DATASET_PATH.name,
                "features_data_version": model_info[
                    "features_data_version"
                ],
                "test_start": test["hour"].min().isoformat(),
                "test_end": test["hour"].max().isoformat(),
                "n_test": len(test),
                "unit_of_analysis": "station-hour",
                "prediction_horizon": "1 hour",
            },
            "evaluation_metadata.json",
        )

        # Métricas en formato CSV
        metrics_df = pd.DataFrame(
            [
                {
                    "dataset": "test",
                    **test_metrics,
                    "n_observations": len(test),
                    "risk_rate": float(y_test.mean()),
                    "risk_threshold": float(risk_threshold),
                }
            ]
        )

        mlflow.log_text(
            metrics_df.to_csv(index=False),
            "test_metrics.csv",
        )

        # Configuración utilizada
        config_path = PROJ_ROOT / "mi_bici" / "config.py"

        if config_path.is_file():
            mlflow.log_artifact(
                str(config_path),
                artifact_path="config",
            )

        logger.success(
            f"Evaluación final P2 registrada. "
            f"MLflow Run ID: {run.info.run_id}"
        )

    # ---------------------------------------------------------
    # 11. Resultado final
    # ---------------------------------------------------------

    logger.success(
        f"TEST Recall: {test_metrics['recall']:.4f}"
    )

    logger.success(
        f"TEST Precision: {test_metrics['precision']:.4f}"
    )

    logger.success(
        f"TEST F1: {test_metrics['f1']:.4f}"
    )

    logger.success(
        f"TEST ROC-AUC: {test_metrics['roc_auc']:.4f}"
    )

    logger.success(
        f"Observaciones TEST: {len(test):,}"
    )


if __name__ == "__main__":
    main()