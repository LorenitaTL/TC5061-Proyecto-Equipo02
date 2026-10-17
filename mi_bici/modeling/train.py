"""
Entrenamiento del modelo P2: clasificación de riesgo operacional.

Pregunta de negocio
-------------------
¿Una estación presentará un nivel alto de desequilibrio operacional
durante la siguiente hora?

Objetivo
--------
Entrenar un modelo de clasificación binaria que permita identificar
estaciones con alto desequilibrio operacional durante la siguiente hora.

La unidad de análisis es estación-hora.

El modelo utiliza información disponible al finalizar la hora t para
predecir el riesgo durante la hora t+1.

El riesgo se define mediante un umbral correspondiente al percentil 80
del desequilibrio operacional normalizado. El umbral se calcula
exclusivamente utilizando el conjunto de entrenamiento para evitar
fuga de información.
"""

import hashlib
from pathlib import Path
import subprocess

import joblib
from loguru import logger
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
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
import typer

from mi_bici.config import (
    IMPUTER_STRATEGY,
    LOGISTIC_CLASS_WEIGHT,
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RANDOM_STATE,
    RISK_PERCENTILE,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)
from mi_bici.modeling.tracking_p1 import TRACKING_URI, get_data_version

app = typer.Typer()


# ---------------------------------------------------------
# Rutas
# ---------------------------------------------------------

MODEL_DIR = PROJ_ROOT / "models"

MODEL_PATH = MODEL_DIR / "p2_risk_model.joblib"
METRICS_PATH = MODEL_DIR / "p2_metrics.csv"
THRESHOLD_PATH = MODEL_DIR / "p2_threshold.txt"


# ---------------------------------------------------------
# Configuración de MLflow
# ---------------------------------------------------------

# Misma convención y misma base de MLflow que P1 (tracking_p1).
EXPERIMENT_NAME = "mibici-equipo02-P2"


# ---------------------------------------------------------
# Pregunta de negocio
# ---------------------------------------------------------

BUSINESS_QUESTION = (
    "¿Una estación presentará un nivel alto de desequilibrio "
    "operacional durante la siguiente hora?"
)


# ---------------------------------------------------------
# Variables utilizadas por el modelo
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


# ---------------------------------------------------------
# Funciones auxiliares para reproducibilidad
# ---------------------------------------------------------


def get_git_commit() -> str:
    """
    Obtiene el commit de Git correspondiente al código utilizado.

    Esto permite relacionar un experimento de MLflow con la versión
    exacta del código que produjo el modelo.
    """

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return "unknown"


def calculate_file_hash(
    path: Path,
) -> str:
    """
    Calcula el hash SHA-256 de un archivo.

    Se utiliza como identificador de la versión exacta del archivo
    de datos utilizado para entrenar el modelo.
    """

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# ---------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------


def load_features(
    input_path: Path,
) -> pd.DataFrame:
    """
    Carga las características generadas para P2.

    También valida que exista la información necesaria para construir
    el objetivo y realizar la división temporal.

    Los datos se ordenan cronológicamente porque P2 es un problema de
    predicción temporal y no se debe mezclar información del futuro
    con información del pasado.
    """

    if not input_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de features: {input_path}")

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

    logger.success(f"Observaciones cargadas: {len(df):,}")

    return df


# ---------------------------------------------------------
# División temporal
# ---------------------------------------------------------


def temporal_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Divide los datos respetando el orden cronológico.

    Distribución:
        70 % entrenamiento
        15 % validación
        15 % prueba

    Justificación
    -------------
    En P2 no se utiliza una división aleatoria porque eso podría permitir
    que información de periodos futuros termine en el conjunto de
    entrenamiento.

    La división temporal simula mejor el escenario real:

        pasado → entrenamiento
        periodo posterior → validación
        periodo más reciente → prueba
    """

    unique_hours = np.sort(df["hour"].unique())

    train_end = int(len(unique_hours) * TRAIN_FRACTION)

    validation_end = int(len(unique_hours) * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train_hours = unique_hours[:train_end]

    validation_hours = unique_hours[train_end:validation_end]

    test_hours = unique_hours[validation_end:]

    train = df[df["hour"].isin(train_hours)].copy()

    validation = df[df["hour"].isin(validation_hours)].copy()

    test = df[df["hour"].isin(test_hours)].copy()

    logger.info(
        f"Train: {len(train):,} observaciones ({train['hour'].min()} → {train['hour'].max()})"
    )

    logger.info(
        f"Validation: {len(validation):,} observaciones "
        f"({validation['hour'].min()} → "
        f"{validation['hour'].max()})"
    )

    logger.info(f"Test: {len(test):,} observaciones ({test['hour'].min()} → {test['hour'].max()})")

    return (
        train,
        validation,
        test,
    )


# ---------------------------------------------------------
# Creación de la etiqueta
# ---------------------------------------------------------


def create_risk_label(
    df: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    """
    Crea la etiqueta binaria Risk.

    Risk = 1
        Cuando el desequilibrio de la siguiente hora es igual o superior
        al umbral definido.

    Risk = 0
        Cuando está por debajo del umbral.

    El umbral se calcula únicamente con TRAIN antes de llamar a esta
    función para validation y test. Esto evita utilizar información
    futura para definir la etiqueta.
    """

    return (df["target_imbalance_ratio"] >= threshold).astype(int)


# ---------------------------------------------------------
# Evaluación
# ---------------------------------------------------------


def evaluate_model(
    model,
    X,
    y,
    dataset_name: str,
) -> dict:
    """
    Calcula las métricas del modelo para un conjunto de datos.

    Métricas
    --------
    accuracy
        Proporción de predicciones correctas.

    precision
        Proporción de estaciones clasificadas como de alto riesgo que
        realmente pertenecen a esa categoría.

    recall
        Proporción de estaciones de alto riesgo que el modelo logra
        identificar correctamente.

    f1
        Media armónica entre precision y recall.

    roc_auc
        Capacidad del modelo para distinguir entre las dos clases
        considerando diferentes umbrales.

    Para P2, Recall es la métrica principal porque perder una estación
    realmente riesgosa puede significar no intervenirla a tiempo.
    """

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

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


# ---------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------


@app.command()
def main(
    input_path: Path = (PROCESSED_DATA_DIR / "features_p2.parquet"),
):
    """
    Entrena el modelo de clasificación de riesgo operacional P2.

    Además de generar los archivos locales del modelo, registra en MLflow:

    - parámetros del experimento;
    - características del conjunto de datos;
    - versión de los datos;
    - commit de Git;
    - pregunta de negocio;
    - métricas de entrenamiento, validación y prueba;
    - archivos generados;
    - modelo entrenado.

    De esta forma el experimento puede ser reproducido y auditado.
    """

    logger.info("Iniciando entrenamiento P2...")

    # -----------------------------------------------------
    # 1. Configuración de MLflow
    # -----------------------------------------------------

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    git_commit = get_git_commit()

    data_version = calculate_file_hash(input_path)
    source_data_version = get_data_version()

    with mlflow.start_run(run_name=("P2-logistic-regression-baseline")) as run:
        # -------------------------------------------------
        # Tags de trazabilidad
        # -------------------------------------------------

        mlflow.set_tags(
            {
                "team": "Equipo02",
                "project": "MiBici",
                "task": "P2",
                "question": "P2",
                "business_question": BUSINESS_QUESTION,
                "git_commit": git_commit,
                "data_version": data_version,
                "prediction_horizon": "1 hour",
                "unit_of_analysis": "station-hour",
            }
        )

        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # -------------------------------------------------
        # 2. Cargar datos
        # -------------------------------------------------

        df = load_features(input_path)

        # El rango temporal real de los datos utilizados se registra
        # después de cargar y validar el archivo.
        data_start = df["hour"].min().isoformat()

        data_end = df["hour"].max().isoformat()

        # -------------------------------------------------
        # 3. División temporal
        # -------------------------------------------------

        (
            train,
            validation,
            test,
        ) = temporal_split(df)

        # -------------------------------------------------
        # 4. Umbral P80 calculado SOLO con train
        # -------------------------------------------------

        risk_threshold = train["target_imbalance_ratio"].quantile(RISK_PERCENTILE)

        logger.info(f"Umbral de riesgo P80 calculado sobre TRAIN: {risk_threshold:.6f}")

        train["Risk"] = create_risk_label(
            train,
            risk_threshold,
        )

        validation["Risk"] = create_risk_label(
            validation,
            risk_threshold,
        )

        test["Risk"] = create_risk_label(
            test,
            risk_threshold,
        )

        # -------------------------------------------------
        # 5. Registrar parámetros
        # -------------------------------------------------

        mlflow.log_params(
            {
                # Modelo
                "model": ("LogisticRegression"),
                "class_weight": LOGISTIC_CLASS_WEIGHT,
                "max_iter": LOGISTIC_MAX_ITER,
                "solver": LOGISTIC_SOLVER,
                "random_state": RANDOM_STATE,
                # Preprocesamiento
                "imputer_strategy": IMPUTER_STRATEGY,
                "scaler": "StandardScaler",
                # Definición del problema
                "risk_percentile": RISK_PERCENTILE,
                "prediction_horizon": "1h",
                # División temporal
                "train_fraction": TRAIN_FRACTION,
                "validation_fraction": VALIDATION_FRACTION,
                "test_fraction": TEST_FRACTION,
                # Características
                "n_features": len(FEATURE_COLUMNS),
                "features": ",".join(FEATURE_COLUMNS),
                # Datos
                "data_start": data_start,
                "data_end": data_end,
                "data_file": input_path.name,
                "data_version": data_version,
                "source_data_version": source_data_version,
                "source_manifest": "references/data_manifest.json",
            }
        )

        # -------------------------------------------------
        # 6. Registrar información del dataset
        # -------------------------------------------------

        mlflow.log_metrics(
            {
                "train_observations": float(len(train)),
                "validation_observations": float(len(validation)),
                "test_observations": float(len(test)),
                "risk_threshold": float(risk_threshold),
                "train_risk_rate": float(train["Risk"].mean()),
                "validation_risk_rate": float(validation["Risk"].mean()),
                "test_risk_rate": float(test["Risk"].mean()),
            }
        )

        # -------------------------------------------------
        # 7. Preparar X / y
        # -------------------------------------------------

        X_train = train[FEATURE_COLUMNS]

        y_train = train["Risk"]

        X_validation = validation[FEATURE_COLUMNS]

        y_validation = validation["Risk"]

        X_test = test[FEATURE_COLUMNS]

        y_test = test["Risk"]

        # -------------------------------------------------
        # 8. Definir modelo
        # -------------------------------------------------

        model = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy=IMPUTER_STRATEGY),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=LOGISTIC_MAX_ITER,
                        class_weight=LOGISTIC_CLASS_WEIGHT,
                        solver=LOGISTIC_SOLVER,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        logger.info("Entrenando Logistic Regression...")

        model.fit(
            X_train,
            y_train,
        )

        logger.success("Modelo entrenado.")

        # -------------------------------------------------
        # 9. Evaluación
        # -------------------------------------------------

        train_metrics = evaluate_model(
            model,
            X_train,
            y_train,
            "train",
        )

        validation_metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
            "validation",
        )

        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
            "test",
        )

        # -------------------------------------------------
        # 10. Registrar métricas en MLflow
        # -------------------------------------------------

        for name, value in train_metrics.items():
            mlflow.log_metric(
                f"train_{name}",
                float(value),
            )

        for name, value in validation_metrics.items():
            mlflow.log_metric(
                f"validation_{name}",
                float(value),
            )

        for name, value in test_metrics.items():
            mlflow.log_metric(
                f"test_{name}",
                float(value),
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

        mlflow.log_artifact(str(METRICS_PATH))

        mlflow.log_artifact(str(THRESHOLD_PATH))

        # -------------------------------------------------
        # 13. Registrar modelo en MLflow
        # -------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="p2_risk_model",
            skops_trusted_types=["numpy.dtype"],
        )

        logger.success("Modelo registrado en MLflow.")

        logger.success(f"MLflow Run ID: {run.info.run_id}")

        logger.success(f"Commit de Git: {git_commit}")

        logger.success(f"Versión de datos: {data_version[:12]}...")

        logger.success(f"Modelo local: {MODEL_PATH}")

        logger.success(f"Métricas: {METRICS_PATH}")

        logger.success(f"Threshold: {THRESHOLD_PATH}")


if __name__ == "__main__":
    app()
