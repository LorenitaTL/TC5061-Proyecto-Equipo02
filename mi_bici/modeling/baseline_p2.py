"""Registro del baseline de persistencia de riesgo operacional P2 en MLflow."""

from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import typer

from mi_bici.config import (
    RISK_PERCENTILE,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from mi_bici.dataset import calculate_sha256
from mi_bici.features_p2 import DATASET_PATH
from mi_bici.modeling.tracking import configure_mlflow, log_traceability
from mi_bici.modeling.train_p2 import (
    create_risk_label,
    load_features,
    temporal_split,
)

app = typer.Typer()


def persistence_prediction(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Predice Risk = 1 si el desbalance de la hora actual alcanza el umbral."""
    return (df["imbalance_ratio"] >= threshold).astype(int)


def persistence_metrics(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Calcula las mismas métricas que la regresión logística de P2."""
    y_true = create_risk_label(df, threshold)
    y_pred = persistence_prediction(df, threshold)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # El desbalance actual es continuo, así que sirve como score para ROC-AUC.
        "roc_auc": float(roc_auc_score(y_true, df["imbalance_ratio"])),
    }


@app.command()
def main(input_path: Path = DATASET_PATH) -> None:
    """Evalúa y registra la regla de persistencia sobre las mismas particiones del modelo."""
    df = load_features(input_path)

    missing = int(df["imbalance_ratio"].isna().sum())
    if missing:
        raise ValueError(f"imbalance_ratio tiene {missing:,} valores faltantes.")

    train, validation, test = temporal_split(df)

    # El umbral se calcula solo con train, igual que en train_p2.py.
    risk_threshold = float(train["target_imbalance_ratio"].quantile(RISK_PERCENTILE))

    metrics = {"train_risk_threshold": risk_threshold}
    for split, data in [("train", train), ("validation", validation), ("test", test)]:
        for name, value in persistence_metrics(data, risk_threshold).items():
            metrics[f"{split}_{name}"] = value
        metrics[f"{split}_observations"] = float(len(data))

    configure_mlflow()

    with mlflow.start_run(run_name="p2-baseline-persistencia-p80") as run:
        log_traceability("P2")
        mlflow.log_params(
            {
                "model": "persistencia",
                "rule": "imbalance_ratio >= train_risk_threshold",
                "risk_percentile": RISK_PERCENTILE,
                "prediction_horizon": "1h",
                "train_fraction": TRAIN_FRACTION,
                "validation_fraction": VALIDATION_FRACTION,
                "test_fraction": TEST_FRACTION,
                "data_start": df["hour"].min().isoformat(),
                "data_end": df["hour"].max().isoformat(),
                "data_file": input_path.name,
                "features_data_version": calculate_sha256(input_path),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {
                "description": (
                    "Baseline de persistencia: predice riesgo en la hora siguiente "
                    "cuando el desbalance relativo de la hora actual alcanza el "
                    "umbral P80 calculado con entrenamiento."
                ),
                "dataset": input_path.name,
                "unit_of_analysis": "estacion-hora",
            },
            "baseline_metadata.json",
        )

        logger.success(f"Run ID: {run.info.run_id}")
        logger.success(f"Umbral P80 (train): {risk_threshold:.6f}")
        for split in ["validation", "test"]:
            logger.success(
                f"{split} | Recall={metrics[f'{split}_recall']:.4f} | "
                f"Precision={metrics[f'{split}_precision']:.4f} | "
                f"F1={metrics[f'{split}_f1']:.4f} | "
                f"ROC-AUC={metrics[f'{split}_roc_auc']:.4f}"
            )


if __name__ == "__main__":
    app()
