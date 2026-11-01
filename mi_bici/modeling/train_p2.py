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

from contextlib import nullcontext
from pathlib import Path

import joblib
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from mi_bici.config import (
    IMPUTER_STRATEGY,
    LOGISTIC_CLASS_WEIGHT,
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
    P2_CLASSIFICATION_THRESHOLD,
    P2_HGB_LEARNING_RATE,
    P2_HGB_MAX_ITER,
    P2_HGB_MAX_LEAF_NODES,
    P2_HGB_MIN_SAMPLES_LEAF,
    P2_MODEL_FAMILY,
    P2_FAMILY_SAMPLE_SIZE,
    PROJ_ROOT,
    RANDOM_STATE,
    RISK_PERCENTILE,
)
from mi_bici.modeling.splits_p2 import TRAIN_END, VALIDATION_END, temporal_masks

from mi_bici.data_tracking import log_training_data
from mi_bici.dataset import calculate_sha256
from mi_bici.features_p2 import DATASET_PATH
from mi_bici.modeling.tracking import configure_mlflow
from mi_bici.modeling.tracking_p2 import log_p2_traceability

app = typer.Typer()


# ---------------------------------------------------------
# Rutas
# ---------------------------------------------------------

MODEL_DIR = PROJ_ROOT / "models"

MODEL_PATH = MODEL_DIR / "p2_risk_model.joblib"
METRICS_PATH = MODEL_DIR / "p2_metrics.csv"
THRESHOLD_PATH = MODEL_DIR / "p2_threshold.txt"


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
# Ablaciones
# ---------------------------------------------------------

# Variables que se quitan en cada versión del modelo. "completo" es el modelo final.
ABLATIONS = {
    "completo": [],
    "sin_desbalance_actual": [
        "absolute_imbalance",
        "imbalance_ratio",
    ],
    "sin_flujo_actual": [
        "departures",
        "arrivals",
        "net_flow",
        "absolute_imbalance",
        "imbalance_ratio",
    ],
}


def ablation_features(ablation: str) -> list[str]:
    """Devuelve las variables de FEATURE_COLUMNS que conserva la ablación."""
    removed = ABLATIONS[ablation]
    return [column for column in FEATURE_COLUMNS if column not in removed]


def ablation_run_name(
    ablation: str,
    model_family: str = P2_MODEL_FAMILY,
) -> str:
    """Construye el nombre del run en MLflow según la familia de modelo."""

    family_name = model_family.replace("_", "-")

    if ablation == "completo":
        return f"P2-{family_name}-candidate"

    return f"P2-{family_name}-{ablation.replace('_', '-')}"

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
    """Divide los datos utilizando las fechas de corte definidas para P2."""

    masks = temporal_masks(df["hour"])

    train = df[masks["train"]].copy()
    validation = df[masks["validation"]].copy()
    test = df[masks["test"]].copy()

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

    return (df["target_imbalance_ratio"] > threshold).astype(int)


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



def build_pipeline(model):
    """
    Construye el pipeline de preprocesamiento y clasificación para P2.

    El clasificador se recibe como argumento para permitir comparar
    diferentes familias de modelos manteniendo el mismo preprocesamiento.
    """
    return Pipeline(
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
                model,
            ),
        ]
    )


def prepare_family_comparison(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    float,
]:
    """
    Prepara los datos para comparar las familias de modelos de P2.

    Utiliza el mismo split temporal y el mismo umbral de riesgo
    para las tres familias.
    """
    train, validation, test = temporal_split(df)

    threshold = train["target_imbalance_ratio"].quantile(RISK_PERCENTILE)

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    train["Risk"] = create_risk_label(train, threshold)
    validation["Risk"] = create_risk_label(validation, threshold)
    test["Risk"] = create_risk_label(test, threshold)

    if len(train) > P2_FAMILY_SAMPLE_SIZE:
        train = train.sample(
            n=P2_FAMILY_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    return train, validation, test, threshold


def get_family_models() -> dict[str, object]:
    """Devuelve los modelos candidatos para la comparación E4."""

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=LOGISTIC_MAX_ITER,
            class_weight=LOGISTIC_CLASS_WEIGHT,
            solver=LOGISTIC_SOLVER,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=P2_HGB_LEARNING_RATE,
            max_iter=P2_HGB_MAX_ITER,
            max_leaf_nodes=P2_HGB_MAX_LEAF_NODES,
            min_samples_leaf=P2_HGB_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }

def get_classifier(model_family: str):
    """Devuelve el clasificador correspondiente a una familia de modelos."""
    models = get_family_models()

    if model_family not in models:
        raise ValueError(
            f"Familia de modelo desconocida: {model_family}. "
            f"Opciones: {list(models)}"
        )

    return models[model_family]


def compare_model_families(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Entrena y compara las familias de modelos de P2.

    La selección del threshold se realiza únicamente sobre validation,
    buscando maximizar F1 sujeto a una precision mínima de 0.50.
    """
    features = ablation_features("completo")

    X_train = train[features]
    y_train = train["Risk"]

    X_validation = validation[features]
    y_validation = validation["Risk"]

    results = []

    for family, classifier in get_family_models().items():
        logger.info(f"Entrenando familia: {family}")

        model = build_pipeline(classifier)
        model.fit(X_train, y_train)

        threshold_results = evaluate_thresholds(
            model,
            X_validation,
            y_validation,
        )

        selected = select_threshold(
            threshold_results,
            min_precision=0.50,
        )

        results.append(
            {
                "model_family": family,
                "threshold": selected["threshold"],
                "accuracy": accuracy_score(
                    y_validation,
                    (
                        model.predict_proba(X_validation)[:, 1]
                        >= selected["threshold"]
                    ).astype(int),
                ),
                "precision": selected["precision"],
                "recall": selected["recall"],
                "f1": selected["f1"],
                "roc_auc": roc_auc_score(
                    y_validation,
                    model.predict_proba(X_validation)[:, 1],
                ),
            }
        )

        logger.info(
            f"{family} | "
            f"Threshold={selected['threshold']:.2f} | "
            f"Recall={selected['recall']:.4f} | "
            f"Precision={selected['precision']:.4f} | "
            f"F1={selected['f1']:.4f}"
        )

    return pd.DataFrame(results).sort_values(
        by="f1",
        ascending=False,
    )


def evaluate_thresholds(
    model,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """
    Evalúa distintos thresholds sobre validation.

    El modelo ya debe estar entrenado. El threshold se selecciona
    únicamente con datos de validation.
    """
    probabilities = model.predict_proba(X_validation)[:, 1]

    results = []

    for threshold in [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(results)

def select_threshold(
    threshold_results: pd.DataFrame,
    min_precision: float = 0.50,
) -> pd.Series:
    """
    Selecciona el threshold con mayor F1 sujeto a una precisión mínima.
    """
    candidates = threshold_results[
        threshold_results["precision"] >= min_precision
    ]

    if candidates.empty:
        raise ValueError(
            f"No hay thresholds con precision >= {min_precision:.2f}"
        )

    return candidates.sort_values(
        by=["f1", "recall"],
        ascending=[False, False],
    ).iloc[0]

def search_hgb_hyperparameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Busca una configuración pequeña de HistGradientBoosting.

    El umbral de riesgo se calcula únicamente sobre TRAIN y se utiliza
    para construir las etiquetas de TRAIN y VALIDATION.

    La selección del threshold de clasificación se realiza únicamente
    sobre VALIDATION, buscando maximizar F1 sujeto a una precision
    mínima de 0.50.
    """

    features = ablation_features("completo")

    train = train.copy()
    validation = validation.copy()

    # ---------------------------------------------------------
    # Crear etiqueta Risk usando únicamente TRAIN
    # ---------------------------------------------------------
    risk_threshold = train["target_imbalance_ratio"].quantile(
        RISK_PERCENTILE
    )

    train["Risk"] = create_risk_label(
        train,
        risk_threshold,
    )

    validation["Risk"] = create_risk_label(
        validation,
        risk_threshold,
    )

    logger.info(
        f"Risk threshold P{RISK_PERCENTILE:.0%}: "
        f"{risk_threshold:.6f}"
    )

    logger.info(
        f"TRAIN positivos: {train['Risk'].mean():.4%}"
    )

    logger.info(
        f"VALIDATION positivos: {validation['Risk'].mean():.4%}"
    )

    X_train = train[features]
    y_train = train["Risk"]

    X_validation = validation[features]
    y_validation = validation["Risk"]

    # ---------------------------------------------------------
    # Búsqueda HGB reducida
    # ---------------------------------------------------------
    param_grid = [
        {
            "learning_rate": 0.10,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        },
        {
            "learning_rate": 0.10,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        },
        {
            "learning_rate": 0.10,
            "max_iter": 100,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
        },
    ]

    results = []

    for i, params in enumerate(param_grid, start=1):
        logger.info(
            f"HGB configuración {i}/{len(param_grid)}: "
            f"{params}"
        )

        classifier = HistGradientBoostingClassifier(
            **params,
            random_state=RANDOM_STATE,
        )

        model = build_pipeline(classifier)

        model.fit(
            X_train,
            y_train,
        )

        threshold_results = evaluate_thresholds(
            model,
            X_validation,
            y_validation,
        )

        selected = select_threshold(
            threshold_results,
            min_precision=0.50,
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        predictions = (
            probabilities >= selected["threshold"]
        ).astype(int)

        results.append(
            {
                **params,
                "threshold": selected["threshold"],
                "accuracy": accuracy_score(
                    y_validation,
                    predictions,
                ),
                "precision": selected["precision"],
                "recall": selected["recall"],
                "f1": selected["f1"],
                "roc_auc": roc_auc_score(
                    y_validation,
                    probabilities,
                ),
            }
        )

        logger.info(
            f"HGB configuración {i} | "
            f"Threshold={selected['threshold']:.2f} | "
            f"Recall={selected['recall']:.4f} | "
            f"Precision={selected['precision']:.4f} | "
            f"F1={selected['f1']:.4f} | "
            f"ROC-AUC={results[-1]['roc_auc']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["f1", "recall", "roc_auc"],
        ascending=[False, False, False],
    )

    logger.success(
        "Mejor configuración HGB: "
        f"{results_df.iloc[0].to_dict()}"
    )

    return results_df.reset_index(drop=True)


def run_hyperparameter_search(
    input_path: Path,
) -> None:
    """
    Ejecuta una búsqueda de hiperparámetros para P2 utilizando
    nested runs de MLflow.

    Una corrida padre representa la búsqueda completa y cada
    combinación de hiperparámetros se registra como una corrida hija.
    """

    logger.info("Iniciando búsqueda de hiperparámetros P2...")

    configure_mlflow()

    features_data_version = calculate_sha256(input_path)

    df = load_features(input_path)

    train, validation, test = temporal_split(df)

    risk_threshold = train["target_imbalance_ratio"].quantile(RISK_PERCENTILE)

    train["Risk"] = create_risk_label(train, risk_threshold)
    validation["Risk"] = create_risk_label(validation, risk_threshold)
    test["Risk"] = create_risk_label(test, risk_threshold)

    search_space = [
        {
            "C": 1.0,
            "class_weight": "balanced",
            "ablation": "completo",
        },
        {
            "C": 1.0,
            "class_weight": None,
            "ablation": "completo",
        },
        {
            "C": 1.0,
            "class_weight": "balanced",
            "ablation": "sin_desbalance_actual",
        },
        {
            "C": 1.0,
            "class_weight": "balanced",
            "ablation": "sin_flujo_actual",
        },
    ]

    parent_run = mlflow.active_run()

    if parent_run is None:
        parent_context = mlflow.start_run(run_name="P2-logistic-regression-search")
    else:
        parent_context = nullcontext(parent_run)

    with parent_context as parent_run:
        mlflow.set_tags(
            {
                "project": "MiBici",
                "question": "P2",
                "model_family": "LogisticRegression",
                "stage_of_work": "search",
                "search_type": "nested",
                "business_question": BUSINESS_QUESTION,
                "prediction_horizon": "1 hour",
                "unit_of_analysis": "station-hour",
                "features_data_version": features_data_version,
                "team": "equipo02",
                "git_commit": "32fb353",
            }
        )

        mlflow.log_params(
            {
                "search_space": (
                    "C=[1.0]; "
                    "class_weight=[balanced,None]; "
                    "ablation=[completo,"
                    "sin_desbalance_actual,"
                    "sin_flujo_actual]"
                ),
                "n_candidates": len(search_space),
                "risk_percentile": RISK_PERCENTILE,
                "class_weight": LOGISTIC_CLASS_WEIGHT,
                "max_iter": LOGISTIC_MAX_ITER,
                "solver": LOGISTIC_SOLVER,
                "imputer_strategy": IMPUTER_STRATEGY,
                "scaler": "StandardScaler",
            }
        )

        logger.info(f"MLflow parent Run ID: {parent_run.info.run_id}")

        for candidate in search_space:
            C = candidate["C"]
            class_weight = candidate["class_weight"]

            class_weight_name = "balanced" if class_weight == "balanced" else "none"

            ablation = candidate["ablation"]
            features = ablation_features(ablation)

            # Features específicas de cada candidato
            X_train = train[features]
            y_train = train["Risk"]

            X_validation = validation[features]
            y_validation = validation["Risk"]

            child_name = f"P2-search-C-{C}-weight-{class_weight_name}-ablation-{ablation}"

            with mlflow.start_run(
                run_name=child_name,
                nested=True,
            ):
                mlflow.set_tags(
                    {
                        "project": "MiBici",
                        "question": "P2",
                        "model_family": "LogisticRegression",
                        "stage_of_work": "candidate",
                        "search_type": "nested",
                        "parent_run_id": parent_run.info.run_id,
                        "team": "equipo02",
                        "git_commit": "32fb353",
                        "ablation": ablation,
                    }
                )

                mlflow.log_params(
                    {
                        "C": C,
                        "class_weight": class_weight,
                        "ablation": ablation,
                        "max_iter": LOGISTIC_MAX_ITER,
                        "solver": LOGISTIC_SOLVER,
                        "imputer_strategy": IMPUTER_STRATEGY,
                        "scaler": "StandardScaler",
                        "risk_percentile": RISK_PERCENTILE,
                        "features_data_version": features_data_version,
                    }
                )

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
                                C=C,
                                max_iter=LOGISTIC_MAX_ITER,
                                class_weight=class_weight,
                                solver=LOGISTIC_SOLVER,
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )

                logger.info(
                    f"Entrenando candidato "
                    f"C={C}, "
                    f"class_weight={class_weight_name}, "
                    f"ablation={ablation}..."
                )

                log_training_data(
                    X_train.assign(Risk=y_train.to_numpy()),
                    name=f"p2_search_{ablation}_{class_weight_name}",
                    source=input_path.resolve().as_uri(),
                    target="Risk",
                    )

                model.fit(X_train, y_train)

                # ---------------------------------------------------------
                # Validation metrics
                # ---------------------------------------------------------

                validation_metrics = evaluate_model(
                    model,
                    X_validation,
                    y_validation,
                    "validation",
                )

                for metric_name, value in validation_metrics.items():
                    mlflow.log_metric(
                        f"validation_{metric_name}",
                        float(value),
                    )

                # ---------------------------------------------------------
                # Validation plots
                # ---------------------------------------------------------

                y_validation_pred = model.predict(X_validation)

                y_validation_proba = model.predict_proba(X_validation)[:, 1]

                # Confusion matrix
                fig, ax = plt.subplots(figsize=(6, 5))

                ConfusionMatrixDisplay.from_predictions(
                    y_validation,
                    y_validation_pred,
                    ax=ax,
                )

                ax.set_title(
                    f"Confusion Matrix - "
                    f"C={C}, "
                    f"class_weight={class_weight_name}, "
                    f"ablation={ablation}"
                )

                mlflow.log_figure(
                    fig,
                    "plots/confusion_matrix_validation.png",
                )

                plt.close(fig)

                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(
                    y_validation,
                    y_validation_proba,
                )

                fig, ax = plt.subplots(figsize=(6, 5))

                ax.plot(
                    recall,
                    precision,
                )

                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")

                ax.set_title(
                    f"Precision-Recall Curve - "
                    f"C={C}, "
                    f"class_weight={class_weight_name}, "
                    f"ablation={ablation}"
                )

                ax.grid(True)

                mlflow.log_figure(
                    fig,
                    "plots/precision_recall_curve_validation.png",
                )

                plt.close(fig)

                # ---------------------------------------------------------
                # Model
                # ---------------------------------------------------------

                input_example = X_validation.head(5)

                signature = infer_signature(
                    X_train,
                    model.predict(X_train),
                )

                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    skops_trusted_types=["numpy.dtype"],
                )

                # ---------------------------------------------------------
                # Configuration artifact
                # ---------------------------------------------------------

                config_path = PROJ_ROOT / "mi_bici" / "config.py"

                if config_path.exists():
                    mlflow.log_artifact(
                        str(config_path),
                        artifact_path="config",
                    )

                # ---------------------------------------------------------
                # Additional traceability information
                # ---------------------------------------------------------

                mlflow.log_metric(
                    "train_observations",
                    float(len(train)),
                )

                mlflow.log_metric(
                    "validation_observations",
                    float(len(validation)),
                )

                mlflow.log_param(
                    "train_risk_threshold",
                    float(risk_threshold),
                )

                logger.success(
                    f"C={C} | "
                    f"class_weight={class_weight_name} | "
                    f"ablation={ablation} | "
                    f"Validation Recall="
                    f"{validation_metrics['recall']:.4f}"
                )

    logger.success("Búsqueda de hiperparámetros P2 terminada.")


# ---------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------


@app.command()
def main(
    input_path: Path = DATASET_PATH,
    ablation: str = typer.Option(
        "completo",
        help=f"Versión a entrenar: {', '.join(ABLATIONS)} o todas",
    ),
):
    """Entrena el modelo completo de P2 o sus ablaciones, un run de MLflow por versión."""
    if ablation == "todas":
        ablations = list(ABLATIONS)
    elif ablation in ABLATIONS:
        ablations = [ablation]
    else:
        raise typer.BadParameter(f"Ablación desconocida: {ablation}")

    for name in ablations:
        train_model(input_path, name)


def train_model(
    input_path: Path,
    ablation: str,
    model_family: str = P2_MODEL_FAMILY,
):
    """
    Entrena el modelo de clasificación de riesgo operacional P2.

    Se distinguen dos umbrales:

    - risk_threshold:
        Percentil P80 del índice target_imbalance_ratio calculado
        exclusivamente sobre TRAIN. Define la etiqueta Risk.

    - classification_threshold:
        Umbral aplicado a la probabilidad predicha por el modelo para
        convertirla en la clase Risk.

    Los hiperparámetros y el threshold de clasificación se obtienen
    desde mi_bici.config para garantizar reproducibilidad.
    """

    logger.info(f"Iniciando entrenamiento P2 ({ablation})...")

    features = ablation_features(ablation)

    classification_threshold = P2_CLASSIFICATION_THRESHOLD

    # -----------------------------------------------------
    # 1. Configuración de MLflow
    # -----------------------------------------------------

    configure_mlflow()

    features_data_version = calculate_sha256(input_path)

    run = mlflow.active_run()

    if run is None:
        run_context = mlflow.start_run(
            run_name=ablation_run_name(
                ablation,
                model_family=model_family,
            )
        )
    else:
        run_context = nullcontext(run)

    with run_context as run:

        # -------------------------------------------------
        # Tags de trazabilidad
        # -------------------------------------------------

        log_p2_traceability(
            model_family=model_family,
            stage_of_work="candidate",
        )

        mlflow.set_tags(
            {
                "project": "MiBici",
                "question": "P2",
                "business_question": BUSINESS_QUESTION,
                "prediction_horizon": "1 hour",
                "unit_of_analysis": "station-hour",
                "ablation": ablation,
            }
        )

        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # -------------------------------------------------
        # 2. Cargar datos
        # -------------------------------------------------

        df = load_features(input_path)

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

        risk_threshold = train[
            "target_imbalance_ratio"
        ].quantile(RISK_PERCENTILE)

        logger.info(
            "Umbral de riesgo P80 calculado sobre TRAIN: "
            f"{risk_threshold:.6f}"
        )

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

        logger.info(
            f"TRAIN positive rate: {train['Risk'].mean():.4%}"
        )

        logger.info(
            f"VALIDATION positive rate: "
            f"{validation['Risk'].mean():.4%}"
        )

        logger.info(
            f"TEST positive rate: {test['Risk'].mean():.4%}"
        )

        log_training_data(
            train[features + ["Risk"]],
            name="p2_training_data",
            source=input_path.resolve().as_uri(),
            target="Risk",
        )

        # -------------------------------------------------
        # 5. Registrar parámetros
        # -------------------------------------------------

        mlflow.log_params(
            {
                # Modelo
                "model_family": model_family,
                "classification_threshold": classification_threshold,
                "random_state": RANDOM_STATE,

                # HistGradientBoosting
                "learning_rate": P2_HGB_LEARNING_RATE,
                "max_iter": P2_HGB_MAX_ITER,
                "max_leaf_nodes": P2_HGB_MAX_LEAF_NODES,
                "min_samples_leaf": P2_HGB_MIN_SAMPLES_LEAF,

                # Preprocesamiento
                "imputer_strategy": IMPUTER_STRATEGY,
                "scaler": "StandardScaler",

                # Definición del problema
                "risk_percentile": RISK_PERCENTILE,
                "risk_threshold": float(risk_threshold),
                "prediction_horizon": "1h",

                # División temporal
                "train_end": TRAIN_END.strftime("%Y-%m-%d"),
                "validation_end": VALIDATION_END.strftime(
                    "%Y-%m-%d"
                ),

                # Características
                "ablation": ablation,
                "removed_features": ",".join(
                    ABLATIONS[ablation]
                ) or "ninguna",
                "n_features": len(features),
                "features": ",".join(features),

                # Datos
                "data_start": data_start,
                "data_end": data_end,
                "data_file": input_path.name,
                "features_data_version": features_data_version,
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
                "train_risk_threshold": float(risk_threshold),
                "train_risk_rate": float(train["Risk"].mean()),
                "validation_risk_rate": float(
                    validation["Risk"].mean()
                ),
                "test_risk_rate": float(
                    test["Risk"].mean()
                ),
            }
        )

        # -------------------------------------------------
        # 7. Preparar X / y
        # -------------------------------------------------

        X_train = train[features]
        y_train = train["Risk"]

        X_validation = validation[features]
        y_validation = validation["Risk"]

        X_test = test[features]
        y_test = test["Risk"]

        # -------------------------------------------------
        # 8. Definir modelo
        # -------------------------------------------------



        classifier = get_classifier(model_family)

        model = build_pipeline(classifier)

        logger.info(
            f"Entrenando {model_family}..."
        )

        model.fit(
            X_train,
            y_train,
        )

        logger.success("Modelo entrenado.")

        # -------------------------------------------------
        # 9. Evaluación con threshold de clasificación
        # -------------------------------------------------

        def evaluate_with_threshold(
            dataset_name: str,
            X: pd.DataFrame,
            y: pd.Series,
        ) -> dict:

            probabilities = model.predict_proba(X)[:, 1]

            predictions = (
                probabilities >= classification_threshold
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
                f"Threshold={classification_threshold:.2f} | "
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )

            return {
                **metrics,
                "predictions": predictions,
                "probabilities": probabilities,
            }

        train_metrics = evaluate_with_threshold(
            "train",
            X_train,
            y_train,
        )

        validation_metrics = evaluate_with_threshold(
            "validation",
            X_validation,
            y_validation,
        )

        test_metrics = evaluate_with_threshold(
            "test",
            X_test,
            y_test,
        )

        # -------------------------------------------------
        # 9.1 Registrar métricas en MLflow
        # -------------------------------------------------

        for name, value in train_metrics.items():
            if isinstance(value, (int, float, np.number)):
                mlflow.log_metric(
                    f"train_{name}",
                    float(value),
                )

        for name, value in validation_metrics.items():
            if isinstance(value, (int, float, np.number)):
                mlflow.log_metric(
                    f"validation_{name}",
                    float(value),
                )

        for name, value in test_metrics.items():
            if isinstance(value, (int, float, np.number)):
                mlflow.log_metric(
                    f"test_{name}",
                    float(value),
                )

        mlflow.log_metric(
            "classification_threshold",
            float(classification_threshold),
        )

        # -------------------------------------------------
        # 9.2 Matriz de confusión para P2
        # -------------------------------------------------

        test_predictions = test_metrics["predictions"]

        confusion_matrix_display = (
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                test_predictions,
                display_labels=[
                    "Riesgo normal",
                    "Alto riesgo",
                ],
                values_format="d",
            )
        )

        confusion_matrix_display.ax_.set_title(
            "P2 - Matriz de confusión en TEST"
        )

        mlflow.log_figure(
            confusion_matrix_display.figure_,
            "plots/confusion_matrix_test.png",
        )

        plt.close(
            confusion_matrix_display.figure_
        )

        # -------------------------------------------------
        # 9.3 Curva Precision-Recall para P2
        # -------------------------------------------------

        test_probabilities = test_metrics["probabilities"]

        precision_values, recall_values, _ = (
            precision_recall_curve(
                y_test,
                test_probabilities,
            )
        )

        fig, ax = plt.subplots()

        ax.plot(
            recall_values,
            precision_values,
        )

        ax.set_title(
            "P2 - Curva Precision-Recall en TEST"
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True)

        mlflow.log_figure(
            fig,
            "plots/precision_recall_curve_test.png",
        )

        plt.close(fig)

        # -------------------------------------------------
        # 10. Construir tabla de métricas
        # -------------------------------------------------

        metrics_df = pd.DataFrame(
            [
                {
                    "dataset": "train",
                    **{
                        key: value
                        for key, value in train_metrics.items()
                        if isinstance(
                            value,
                            (int, float, np.number),
                        )
                    },
                },
                {
                    "dataset": "validation",
                    **{
                        key: value
                        for key, value in validation_metrics.items()
                        if isinstance(
                            value,
                            (int, float, np.number),
                        )
                    },
                },
                {
                    "dataset": "test",
                    **{
                        key: value
                        for key, value in test_metrics.items()
                        if isinstance(
                            value,
                            (int, float, np.number),
                        )
                    },
                },
            ]
        )

        # -------------------------------------------------
        # 11. Guardar archivos locales
        # -------------------------------------------------

        classification_threshold_path = (
            MODEL_DIR / "p2_classification_threshold.txt"
        )

        if ablation == "completo":
            MODEL_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            metrics_df.to_csv(
                METRICS_PATH,
                index=False,
            )

            THRESHOLD_PATH.write_text(
                str(risk_threshold),
                encoding="utf-8",
            )

            classification_threshold_path.write_text(
                str(classification_threshold),
                encoding="utf-8",
            )

            joblib.dump(
                model,
                MODEL_PATH,
            )

        # -------------------------------------------------
        # 12. Registrar artifacts en MLflow
        # -------------------------------------------------

        mlflow.log_text(
            metrics_df.to_csv(index=False),
            METRICS_PATH.name,
        )

        mlflow.log_text(
            str(risk_threshold),
            THRESHOLD_PATH.name,
        )

        mlflow.log_text(
            str(classification_threshold),
            classification_threshold_path.name,
        )

        # -------------------------------------------------
        # 12.1 Registrar manifiesto de datos
        # -------------------------------------------------

        data_manifest_path = (
            PROJ_ROOT
            / "references"
            / "data_manifest.json"
        )

        if data_manifest_path.is_file():
            mlflow.log_artifact(
                str(data_manifest_path),
                artifact_path="data",
            )
        else:
            logger.warning(
                "No se encontró el manifiesto de datos: "
                f"{data_manifest_path}"
            )

        # -------------------------------------------------
        # 12.2 Registrar configuración
        # -------------------------------------------------

        config_path = PROJ_ROOT / "mi_bici" / "config.py"

        if config_path.is_file():
            mlflow.log_artifact(
                str(config_path),
                artifact_path="config",
            )
        else:
            logger.warning(
                "No se encontró el archivo de configuración: "
                f"{config_path}"
            )

        # -------------------------------------------------
        # 13. Registrar modelo en MLflow
        # -------------------------------------------------

        signature = infer_signature(
            X_test,
            test_predictions,
        )

        mlflow.sklearn.log_model(
            model,
            name="p2_risk_model",
            signature=signature,
            input_example=X_test.head(5),
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            ],
        )

        # -------------------------------------------------
        # 13.1 Finalizar
        # -------------------------------------------------

        logger.success(
            "Modelo registrado en MLflow."
        )

        logger.success(
            f"MLflow Run ID: {run.info.run_id}"
        )

        logger.success(
            "Versión del archivo de features: "
            f"{features_data_version[:12]}..."
        )

        if ablation == "completo":
            logger.success(
                f"Modelo local: {MODEL_PATH}"
            )

            logger.success(
                f"Métricas: {METRICS_PATH}"
            )

            logger.success(
                f"Risk threshold: {THRESHOLD_PATH}"
            )

            logger.success(
                "Classification threshold: "
                f"{classification_threshold_path}"
            )

def run(model: str, params: dict):
    """
    Entry point for P2 training.

    model:
        - "default" -> familia seleccionada en P2_MODEL_FAMILY.
        - "logistic_regression" -> Logistic Regression.
        - "hist_gradient_boosting" -> HistGradientBoosting.
        - "random_forest" -> Random Forest.
        - "search" -> búsqueda de hiperparámetros de Logistic Regression.
    """

    input_path = Path(params.get("input_path", DATASET_PATH))
    ablation = params.get("ablation", "completo")

    # -------------------------------------------------
    # Búsqueda de hiperparámetros existente
    # -------------------------------------------------
    if model == "search":
        return run_hyperparameter_search(input_path)

    # -------------------------------------------------
    # Selección de familia de modelo
    # -------------------------------------------------
    model_families = {
        "logistic_regression": "LogisticRegression",
        "hist_gradient_boosting": "HistGradientBoosting",
        "random_forest": "RandomForest",
    }

    if model == "default":
        selected_model_family = P2_MODEL_FAMILY
    elif model in model_families:
        selected_model_family = model_families[model]
    else:
        raise ValueError(
            f"Modelo desconocido: {model}. "
            f"Opciones: default, "
            f"{', '.join(model_families)}, search"
        )

    # -------------------------------------------------
    # Normal P2 training
    # -------------------------------------------------
    if ablation == "todas":
        ablations = list(ABLATIONS)
    elif ablation in ABLATIONS:
        ablations = [ablation]
    else:
        raise ValueError(
            f"Ablación desconocida: {ablation}. "
            f"Opciones: {list(ABLATIONS)}"
        )

    run_ids = []

    for name in ablations:
        run_id = train_model(
            input_path,
            name,
            model_family=selected_model_family,
        )
        run_ids.append(run_id)

    return run_ids


if __name__ == "__main__":
    app()
