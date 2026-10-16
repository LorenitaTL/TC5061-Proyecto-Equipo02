from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici.config import PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MODEL_DIR = PROJ_ROOT / "models"
MODEL_PATH = MODEL_DIR / "p2_risk_model.joblib"
METRICS_PATH = MODEL_DIR / "p2_metrics.csv"
THRESHOLD_PATH = MODEL_DIR / "p2_threshold.txt"

# ---------------------------------------------------------
# MLflow
# ---------------------------------------------------------

EXPERIMENT_NAME = "P2-operational-risk"

# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "departures",
    "arrivals",
    "net_flow",
    "absolute_imbalance",
    "imbalance_ratio",
    "capacity",
    "latitude",
    "longitude",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "departures_lag_1h",
    "departures_lag_2h",
    "departures_lag_24h",
    "departures_rolling_3h",
    "departures_rolling_24h",
    "arrivals_lag_1h",
    "arrivals_lag_2h",
    "arrivals_lag_24h",
    "arrivals_rolling_3h",
    "arrivals_rolling_24h",
    "net_flow_lag_1h",
    "net_flow_lag_2h",
    "net_flow_lag_24h",
    "net_flow_rolling_3h",
    "net_flow_rolling_24h",
]


def load_features(input_path: Path) -> pd.DataFrame:
    """Carga las features generadas para P2."""

    if not input_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de features: {input_path}"
        )

    logger.info(f"Leyendo {input_path.name}")

    df = pd.read_parquet(input_path)

    df["hour"] = pd.to_datetime(
        df["hour"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "hour",
            "target_imbalance_ratio",
        ]
    ).copy()

    df = df.sort_values("hour").reset_index(drop=True)

    logger.success(
        f"Observaciones cargadas: {len(df):,}"
    )

    return df


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos cronológicamente:
    70% train, 15% validation, 15% test.
    """

    unique_hours = np.sort(
        df["hour"].unique()
    )

    train_end = int(
        len(unique_hours) * 0.70
    )

    validation_end = int(
        len(unique_hours) * 0.85
    )

    train_hours = unique_hours[:train_end]

    validation_hours = unique_hours[
        train_end:validation_end
    ]

    test_hours = unique_hours[
        validation_end:
    ]

    train = df[
        df["hour"].isin(train_hours)
    ].copy()

    validation = df[
        df["hour"].isin(validation_hours)
    ].copy()

    test = df[
        df["hour"].isin(test_hours)
    ].copy()

    logger.info(
        f"Train: {len(train):,} observaciones "
        f"({train['hour'].min()} → {train['hour'].max()})"
    )

    logger.info(
        f"Validation: {len(validation):,} observaciones "
        f"({validation['hour'].min()} → {validation['hour'].max()})"
    )

    logger.info(
        f"Test: {len(test):,} observaciones "
        f"({test['hour'].min()} → {test['hour'].max()})"
    )

    return train, validation, test


def create_risk_label(
    df: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    """Crea Risk=1 si el desequilibrio supera el umbral."""

    return (
        df["target_imbalance_ratio"] >= threshold
    ).astype(int)


def evaluate_model(
    model,
    X,
    y,
    dataset_name: str,
) -> dict:

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
    }

    logger.info(
        f"{dataset_name} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f}"
    )

    return metrics


@app.command()
def main(
    input_path: Path = (
        PROCESSED_DATA_DIR
        / "features_p2.parquet"
    ),
):
    """
    Entrena el modelo de clasificación
    de riesgo operacional P2 y registra
    el experimento en MLflow.
    """

    logger.info(
        "Iniciando entrenamiento P2..."
    )

    # -----------------------------------------------------
    # 1. Configurar MLflow
    # -----------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(run_name="baseline_all_features") as run:

        logger.info(
            f"MLflow Run ID: {run.info.run_id}"
        )

        # -------------------------------------------------
        # 2. Cargar features
        # -------------------------------------------------

        df = load_features(
            input_path
        )

        # -------------------------------------------------
        # 3. Split temporal
        # -------------------------------------------------

        train, validation, test = (
            temporal_split(df)
        )

        # -------------------------------------------------
        # 4. Threshold P80 SOLO usando train
        # -------------------------------------------------

        risk_threshold = (
            train[
                "target_imbalance_ratio"
            ].quantile(0.80)
        )

        logger.info(
            "Umbral de riesgo P80 calculado "
            f"sobre TRAIN: {risk_threshold:.6f}"
        )

        train["Risk"] = (
            create_risk_label(
                train,
                risk_threshold,
            )
        )

        validation["Risk"] = (
            create_risk_label(
                validation,
                risk_threshold,
            )
        )

        test["Risk"] = (
            create_risk_label(
                test,
                risk_threshold,
            )
        )

        # -------------------------------------------------
        # 5. Log parámetros del experimento
        # -------------------------------------------------

        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 1000,
                "solver": "lbfgs",
                "risk_percentile": 0.80,
                "train_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "n_features": len(
                    FEATURE_COLUMNS
                ),
                "features": ",".join(FEATURE_COLUMNS)
            }
        )

        # -------------------------------------------------
        # 6. Log información del dataset
        # -------------------------------------------------

        mlflow.log_metrics(
            {
                "train_observations": len(train),
                "validation_observations": len(
                    validation
                ),
                "test_observations": len(test),
                "risk_threshold": risk_threshold,
                "train_risk_rate": train[
                    "Risk"
                ].mean(),
                "validation_risk_rate": validation[
                    "Risk"
                ].mean(),
                "test_risk_rate": test[
                    "Risk"
                ].mean(),
            }
        )

        # -------------------------------------------------
        # 7. Preparar X / y
        # -------------------------------------------------

        X_train = train[
            FEATURE_COLUMNS
        ]

        y_train = train["Risk"]

        X_validation = validation[
            FEATURE_COLUMNS
        ]

        y_validation = validation[
            "Risk"
        ]

        X_test = test[
            FEATURE_COLUMNS
        ]

        y_test = test["Risk"]

        # -------------------------------------------------
        # 8. Modelo
        # -------------------------------------------------

        model = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        logger.info(
            "Entrenando Logistic Regression..."
        )

        model.fit(
            X_train,
            y_train,
        )

        logger.success(
            "Modelo entrenado."
        )

        # -------------------------------------------------
        # 9. Evaluación
        # -------------------------------------------------

        train_metrics = (
            evaluate_model(
                model,
                X_train,
                y_train,
                "train",
            )
        )

        validation_metrics = (
            evaluate_model(
                model,
                X_validation,
                y_validation,
                "validation",
            )
        )

        test_metrics = (
            evaluate_model(
                model,
                X_test,
                y_test,
                "test",
            )
        )

        # -------------------------------------------------
        # 10. Registrar métricas en MLflow
        # -------------------------------------------------

        for name, value in train_metrics.items():
            mlflow.log_metric(
                f"train_{name}",
                value,
            )

        for name, value in validation_metrics.items():
            mlflow.log_metric(
                f"validation_{name}",
                value,
            )

        for name, value in test_metrics.items():
            mlflow.log_metric(
                f"test_{name}",
                value,
            )

        # -------------------------------------------------
        # 11. Guardar archivos locales
        # -------------------------------------------------

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_df = pd.DataFrame(
            [
                {
                    "dataset": "train",
                    **train_metrics,
                },
                {
                    "dataset": "validation",
                    **validation_metrics,
                },
                {
                    "dataset": "test",
                    **test_metrics,
                },
            ]
        )

        metrics_df.to_csv(
            METRICS_PATH,
            index=False,
        )

        THRESHOLD_PATH.write_text(
            str(risk_threshold),
            encoding="utf-8",
        )

        joblib.dump(
            model,
            MODEL_PATH,
        )

        # -------------------------------------------------
        # 12. Registrar artifacts en MLflow
        # -------------------------------------------------

        mlflow.log_artifact(
            str(METRICS_PATH)
        )

        mlflow.log_artifact(
            str(THRESHOLD_PATH)
        )

        # -------------------------------------------------
        # 13. Registrar modelo en MLflow
        # -------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="p2_risk_model",
            skops_trusted_types=["numpy.dtype"],
        )

        logger.success(
            "Modelo registrado en MLflow."
        )

        logger.success(
            f"MLflow Run ID: {run.info.run_id}"
        )

        logger.success(
            f"Modelo local: {MODEL_PATH}"
        )

        logger.success(
            f"Métricas: {METRICS_PATH}"
        )

        logger.success(
            f"Threshold: {THRESHOLD_PATH}"
        )


if __name__ == "__main__":
    app()