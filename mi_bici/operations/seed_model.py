"""Modelo semilla de P2 para el sistema en Docker Compose y para el CI (L2).

``mlflow.db`` y ``mlartifacts/`` no están en Git, así que en un clon limpio o en GitHub
Actions no hay ningún modelo registrado y la API arranca sin modelo. Este módulo entrena un
modelo de P2 con datos sintéticos y lo registra como
``mibici-equipo02-p2-riesgo@champion`` en el MLflow que indique ``--tracking-uri``.

Es un modelo para **operar y medir la carga**, no para decidir: tiene las mismas columnas,
el mismo pipeline y los mismos hiperparámetros que el champion real, así que la latencia y
el consumo de recursos son representativos, pero sus predicciones no lo son. La versión
registrada lleva la etiqueta ``origen=seed-sintetico`` para no confundirla con el modelo real.

Uso::

    python -m mi_bici.operations.seed_model                     # MLFLOW_TRACKING_URI
    python -m mi_bici.operations.seed_model --tracking-uri http://mlflow:5000

Es idempotente: si el alias ya existe (por ejemplo, porque se montó el registro real), no
hace nada. ``--force`` registra una versión nueva y le mueve el alias.
"""

from dataclasses import dataclass

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
import typer

from mi_bici.config import (
    MLFLOW_TRACKING_URI,
    P2_CLASSIFICATION_THRESHOLD,
    P2_HGB_LEARNING_RATE,
    P2_HGB_MAX_ITER,
    P2_HGB_MAX_LEAF_NODES,
    P2_HGB_MIN_SAMPLES_LEAF,
    RANDOM_STATE,
)
from mi_bici.modeling.register_p2 import ALIAS, MODEL_ARTIFACT, MODEL_NAME
from mi_bici.modeling.train_p2 import FEATURE_COLUMNS, build_pipeline

SEED_EXPERIMENT = "mibici-equipo02-seed"
SEED_ORIGIN_TAG = ("origen", "seed-sintetico")
DEFAULT_ROWS = 5_000

# Coordenadas aproximadas del área de servicio de MiBici (Guadalajara).
LATITUDE_RANGE = (20.62, 20.74)
LONGITUDE_RANGE = (-103.42, -103.29)

# Tipos que skops debe aceptar al cargar el pipeline (los mismos que usa train_p2).
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

app = typer.Typer()


@dataclass(frozen=True)
class SeedResult:
    """Versión que quedó con el alias y si este llamado la registró."""

    version: str
    created: bool


def _hourly_rate(hours: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Viajes esperados por hora: picos en la mañana y la tarde, menos en fin de semana."""

    morning = np.exp(-0.5 * ((hours - 8) / 1.5) ** 2)
    evening = np.exp(-0.5 * ((hours - 18) / 2.0) ** 2)
    rate = 0.3 + 4.0 * morning + 3.5 * evening
    return np.where(weekend == 1, rate * 0.6, rate)


def make_synthetic_features(n_rows: int = DEFAULT_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Genera ``n_rows`` filas estación-hora con las columnas ``FEATURE_COLUMNS``.

    Las variables respetan las mismas relaciones que ``features_p2``: ``net_flow`` es
    salidas menos llegadas, ``absolute_imbalance`` su valor absoluto e ``imbalance_ratio``
    ese desbalance entre la capacidad. Los rezagos y promedios se generan alrededor de la
    misma tasa horaria, así que los valores son plausibles aunque no sean reales.
    """

    if n_rows <= 0:
        raise ValueError("n_rows debe ser positivo.")

    rng = np.random.default_rng(seed)

    hour = rng.integers(0, 24, n_rows)
    dow = rng.integers(0, 7, n_rows)
    weekend = (dow >= 5).astype(int)
    capacity = rng.integers(10, 31, n_rows)
    # Cada estación tiene un sesgo propio: unas atraen viajes y otras los generan.
    bias = rng.normal(0, 0.6, n_rows)
    rate = _hourly_rate(hour, weekend)

    def counts(scale: np.ndarray) -> np.ndarray:
        return rng.poisson(np.clip(scale, 0.05, None)).astype(float)

    def rolling(scale: np.ndarray, window: int) -> np.ndarray:
        return np.round(rng.poisson(np.clip(scale, 0.05, None) * window) / window, 4)

    dep_scale = rate * np.exp(bias / 2)
    arr_scale = rate * np.exp(-bias / 2)

    data = {
        "departures": counts(dep_scale),
        "arrivals": counts(arr_scale),
    }
    data["net_flow"] = data["departures"] - data["arrivals"]
    data["absolute_imbalance"] = np.abs(data["net_flow"])
    data["imbalance_ratio"] = np.round(data["absolute_imbalance"] / capacity, 4)
    data["capacity"] = capacity
    data["latitude"] = np.round(rng.uniform(*LATITUDE_RANGE, n_rows), 6)
    data["longitude"] = np.round(rng.uniform(*LONGITUDE_RANGE, n_rows), 6)
    data["hour_of_day"] = hour
    data["day_of_week"] = dow
    data["is_weekend"] = weekend
    data["hour_sin"] = np.round(np.sin(2 * np.pi * hour / 24), 4)
    data["hour_cos"] = np.round(np.cos(2 * np.pi * hour / 24), 4)
    data["dow_sin"] = np.round(np.sin(2 * np.pi * dow / 7), 4)
    data["dow_cos"] = np.round(np.cos(2 * np.pi * dow / 7), 4)

    for prefix, scale in (("departures", dep_scale), ("arrivals", arr_scale)):
        data[f"{prefix}_lag_1h"] = counts(scale)
        data[f"{prefix}_lag_2h"] = counts(scale)
        data[f"{prefix}_lag_24h"] = counts(scale)
        data[f"{prefix}_rolling_3h"] = rolling(scale, 3)
        data[f"{prefix}_rolling_24h"] = rolling(scale, 24)

    for suffix in ("lag_1h", "lag_2h", "lag_24h", "rolling_3h", "rolling_24h"):
        data[f"net_flow_{suffix}"] = np.round(
            data[f"departures_{suffix}"] - data[f"arrivals_{suffix}"], 4
        )

    return pd.DataFrame(data)[FEATURE_COLUMNS]


def make_synthetic_labels(features: pd.DataFrame, seed: int = RANDOM_STATE) -> np.ndarray:
    """Etiqueta de riesgo sintética: más probable con desbalance actual y reciente alto.

    La proporción de positivos queda cerca del 20 %, como en P2 (percentil 80).
    """

    rng = np.random.default_rng(seed + 1)
    score = (
        6.0 * features["imbalance_ratio"]
        + 0.35 * features["net_flow_rolling_3h"].abs()
        + 0.25 * features["net_flow_lag_1h"].abs()
        + rng.normal(0, 0.5, len(features))
    )
    return (score >= np.quantile(score, 0.80)).astype(int).to_numpy()


def train_seed_model(features: pd.DataFrame, labels: np.ndarray) -> Pipeline:
    """Entrena el pipeline de P2 con los hiperparámetros del champion real."""

    classifier = HistGradientBoostingClassifier(
        learning_rate=P2_HGB_LEARNING_RATE,
        max_iter=P2_HGB_MAX_ITER,
        max_leaf_nodes=P2_HGB_MAX_LEAF_NODES,
        min_samples_leaf=P2_HGB_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    model = build_pipeline(classifier)
    model.fit(features, labels)
    return model


def current_alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    """Versión a la que apunta ``alias``, o ``None`` si el modelo o el alias no existen."""

    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def register_seed_model(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    n_rows: int = DEFAULT_ROWS,
    force: bool = False,
    artifact_location: str | None = None,
    name: str = MODEL_NAME,
    alias: str = ALIAS,
) -> SeedResult:
    """Entrena y registra el modelo semilla, salvo que el alias ya exista.

    ``artifact_location`` solo se usa al crear el experimento; con un servidor que sirve
    artefactos (``--serve-artifacts``) se deja vacío y el servidor decide dónde guardarlos.
    """

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    existing = current_alias_version(client, name, alias)
    if existing is not None and not force:
        logger.info(f"{name}@{alias} ya existe (versión {existing}); no se registra la semilla.")
        return SeedResult(version=existing, created=False)

    if mlflow.get_experiment_by_name(SEED_EXPERIMENT) is None:
        mlflow.create_experiment(SEED_EXPERIMENT, artifact_location=artifact_location)
    mlflow.set_experiment(SEED_EXPERIMENT)

    features = make_synthetic_features(n_rows)
    labels = make_synthetic_labels(features)
    model = train_seed_model(features, labels)

    with mlflow.start_run(run_name="p2-seed-sintetico"):
        mlflow.set_tags(
            {
                SEED_ORIGIN_TAG[0]: SEED_ORIGIN_TAG[1],
                "question": "P2",
                "stage_of_work": "seed",
                "model_family": "HistGradientBoosting",
            }
        )
        mlflow.log_params(
            {
                "classification_threshold": P2_CLASSIFICATION_THRESHOLD,
                "n_rows": n_rows,
                "learning_rate": P2_HGB_LEARNING_RATE,
                "max_iter": P2_HGB_MAX_ITER,
                "max_leaf_nodes": P2_HGB_MAX_LEAF_NODES,
                "min_samples_leaf": P2_HGB_MIN_SAMPLES_LEAF,
            }
        )
        example = features.head(5)
        model_info = mlflow.sklearn.log_model(
            model,
            name=MODEL_ARTIFACT,
            signature=infer_signature(example, model.predict_proba(example)),
            input_example=example,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    # En MLflow 3 el modelo es un LoggedModel (models:/m-...), no un artefacto del run.
    version = str(mlflow.register_model(model_info.model_uri, name).version)
    client.set_model_version_tag(name, version, *SEED_ORIGIN_TAG)
    client.update_model_version(
        name,
        version,
        description=(
            "Modelo semilla de P2 entrenado con datos sintéticos (mi_bici/operations/"
            "seed_model.py). Sirve para operar el sistema y medir la carga; no para decidir."
        ),
    )
    client.set_registered_model_alias(name, alias, version)
    logger.success(f"Modelo semilla registrado: {name} versión {version} con alias {alias}.")
    return SeedResult(version=version, created=True)


@app.command()
def main(
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI, help="Servidor o base de MLflow."),
    n_rows: int = typer.Option(DEFAULT_ROWS, help="Filas sintéticas de entrenamiento."),
    force: bool = typer.Option(False, help="Registrar aunque el alias ya exista."),
) -> None:
    """Registra el modelo semilla de P2 si el alias champion no existe."""

    register_seed_model(tracking_uri=tracking_uri, n_rows=n_rows, force=force)


if __name__ == "__main__":
    app()
