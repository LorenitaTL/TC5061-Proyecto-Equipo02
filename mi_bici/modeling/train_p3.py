"""Entrenamiento y registro en MLflow de P3: clustering de estaciones por perfil de uso.

``run(model, params)`` es el adaptador del punto de entrada ``train`` del MLflow Project
(``mlflow run . -e train -P question=p3 -P model=kmeans_perfil -P n_clusters=4``). Cada
ejecución registra un run con uno de estos modelos:

- ``kmeans_ubicacion`` (baseline): K-Means sobre latitud y longitud. Responde si basta la
  ubicación para separar estaciones con patrones de uso distintos.
- ``kmeans_perfil`` (candidato, por defecto): K-Means sobre los balances de flujo en horas
  pico y el uso en fin de semana (modelo elegido en
  ``notebooks/1.0-lt-perfiles-estaciones.ipynb``).

Hiperparámetros aceptados: ``n_clusters``, ``n_init`` y ``features`` (lista separada por
comas de ``PROFILE_FEATURES``).

Los modelos se ajustan con el periodo de entrenamiento (DATA_START_MONTH a TRAIN_END_MONTH)
y se evalúan en el espacio de perfiles de uso, para que sus métricas sean comparables. Los
meses posteriores se usan solo como prueba: se asignan con el modelo ya ajustado y se compara
contra un reajuste para medir si los grupos se sostienen en el tiempo.
"""

from loguru import logger
from matplotlib.figure import Figure
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mi_bici.config import (
    DATA_END_MONTH,
    DATA_START_MONTH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TRAIN_END_MONTH,
)
from mi_bici.data_tracking import log_training_data
from mi_bici.features import month_range
from mi_bici.features_p3 import (
    LOCATION_FEATURES,
    MIN_DEPARTURES,
    build_station_profiles,
    load_profile_trips,
    load_station_coordinates,
)
from mi_bici.modeling.tracking import configure_mlflow, log_run_config, log_traceability

N_CLUSTERS = 3
N_INIT = 10
N_STABILITY_SEEDS = 5
WEEKEND_CLIP_QUANTILE = 0.99

USAGE_FEATURES = ["balance_am", "balance_pm", "share_fin_semana"]
PROFILE_FEATURES = [
    "balance_am",
    "balance_pm",
    "balance_dia",
    "share_fin_semana",
    "log_viajes_dia",
    *LOCATION_FEATURES,
]

MODELS = {
    "kmeans_ubicacion": {"features": LOCATION_FEATURES, "stage_of_work": "baseline"},
    "kmeans_perfil": {"features": USAGE_FEATURES, "stage_of_work": "candidate"},
}
DEFAULT_MODEL = "kmeans_perfil"


def resolve_config(model: str, params: dict[str, str]) -> dict:
    """Valida el modelo y los hiperparámetros recibidos como texto y aplica los valores por
    defecto."""
    model_name = DEFAULT_MODEL if model.lower() == "default" else model.lower()
    if model_name not in MODELS:
        raise ValueError(f"Modelo P3 desconocido: {model}. Usa uno de {sorted(MODELS)}.")

    unknown = set(params) - {"n_clusters", "n_init", "features"}
    if unknown:
        raise ValueError(f"Parámetros P3 desconocidos: {sorted(unknown)}")

    config = {
        "model": model_name,
        "features": list(MODELS[model_name]["features"]),
        "stage_of_work": MODELS[model_name]["stage_of_work"],
        "n_clusters": N_CLUSTERS,
        "n_init": N_INIT,
    }
    for key in ("n_clusters", "n_init"):
        if key in params:
            try:
                config[key] = int(params[key])
            except ValueError as error:
                raise ValueError(f"Valor inválido para {key}: {params[key]}") from error
    if config["n_clusters"] < 2 or config["n_init"] < 1:
        raise ValueError("n_clusters debe ser al menos 2 y n_init al menos 1.")

    if "features" in params:
        features = [name.strip() for name in params["features"].split(",") if name.strip()]
        invalid = set(features) - set(PROFILE_FEATURES)
        if not features or invalid:
            raise ValueError(f"features inválidas: {params['features']}. Usa {PROFILE_FEATURES}.")
        config["features"] = features

    return config


def run_name(config: dict) -> str:
    """Nombre descriptivo del run, p. ej. ``p3-kmeans-perfil-k4``."""
    name = f"p3-{config['model'].replace('_', '-')}-k{config['n_clusters']}"
    if config["features"] != MODELS[config["model"]]["features"]:
        name += "-" + "+".join(config["features"])
    return name


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    n_init: int = N_INIT,
    seed: int = RANDOM_STATE,
) -> KMeans:
    """Ajusta K-Means con el número de clusters y reinicios indicados."""
    return KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit(X)


def stability_ari(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    n_init: int = N_INIT,
    seed: int = RANDOM_STATE,
) -> float:
    """ARI promedio entre el ajuste con ``seed`` y ajustes con otras semillas.

    Vale 1 cuando los grupos no dependen de la inicialización. El número de cada cluster sí
    puede cambiar entre semillas; el ARI no se ve afectado por eso.
    """
    reference = fit_kmeans(X, n_clusters, n_init, seed).labels_
    return float(
        np.mean(
            [
                adjusted_rand_score(reference, fit_kmeans(X, n_clusters, n_init, seed + i).labels_)
                for i in range(1, N_STABILITY_SEEDS)
            ]
        )
    )


def cluster_summary(profiles: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Resume cada cluster con su tamaño y el promedio de las variables de perfil."""
    summary = (
        profiles.assign(cluster=labels)
        .groupby("cluster")
        .agg(
            estaciones=("balance_am", "size"),
            balance_am=("balance_am", "mean"),
            balance_pm=("balance_pm", "mean"),
            balance_dia=("balance_dia", "mean"),
            share_fin_semana=("share_fin_semana", "mean"),
            viajes_dia_mediana=("log_viajes_dia", lambda s: np.expm1(s.median())),
        )
    )
    return summary.round(3)


def plot_clusters(profiles: pd.DataFrame, labels: np.ndarray, title: str) -> Figure:
    """Dibuja los clusters en el mapa (izquierda) y en el plano de balances AM/PM (derecha)."""
    fig = Figure(figsize=(13, 6))
    ax_map, ax_balance = fig.subplots(1, 2)

    for cluster in np.unique(labels):
        members = profiles.loc[labels == cluster]
        label = f"Cluster {cluster} ({len(members)})"
        ax_map.scatter(members["longitude"], members["latitude"], s=12, label=label)
        ax_balance.scatter(members["balance_am"], members["balance_pm"], s=12, label=label)

    # Corrige la proporción del mapa a la latitud de Guadalajara.
    ax_map.set_aspect(1 / np.cos(np.radians(profiles["latitude"].mean())))
    ax_map.set(title="Ubicación", xlabel="Longitud", ylabel="Latitud")
    ax_balance.axhline(0, color="grey", lw=0.5)
    ax_balance.axvline(0, color="grey", lw=0.5)
    ax_balance.set(title="Perfil de uso", xlabel="balance_am", ylabel="balance_pm")
    ax_balance.legend()
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def load_profiles() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str | float | int]]:
    """Construye los perfiles de entrenamiento y prueba con sus coordenadas.

    Devuelve también los parámetros de la partición, para registrarlos en cada run.
    """
    test_start_month = str(pd.Period(TRAIN_END_MONTH, freq="M") + 1)
    train_months = len(month_range(DATA_START_MONTH, TRAIN_END_MONTH))
    test_months = len(month_range(test_start_month, DATA_END_MONTH))

    # El mínimo de salidas se ajusta a la duración de cada periodo.
    test_min_departures = round(MIN_DEPARTURES * test_months / train_months)

    coordinates = load_station_coordinates()
    train = build_station_profiles(load_profile_trips(DATA_START_MONTH, TRAIN_END_MONTH))
    test = build_station_profiles(
        load_profile_trips(test_start_month, DATA_END_MONTH),
        min_departures=test_min_departures,
    )
    train = train.join(coordinates, how="inner")
    test = test.join(coordinates, how="inner")

    # Recorta la cola de share_fin_semana con un umbral calculado solo en entrenamiento,
    # para que no forme clusters de una o dos estaciones.
    weekend_clip = float(train["share_fin_semana"].quantile(WEEKEND_CLIP_QUANTILE))
    train["share_fin_semana"] = train["share_fin_semana"].clip(upper=weekend_clip)
    test["share_fin_semana"] = test["share_fin_semana"].clip(upper=weekend_clip)

    split_params = {
        "train_start_month": DATA_START_MONTH,
        "train_end_month": TRAIN_END_MONTH,
        "test_start_month": test_start_month,
        "test_end_month": DATA_END_MONTH,
        "train_min_departures": MIN_DEPARTURES,
        "test_min_departures": test_min_departures,
        "weekend_clip_quantile": WEEKEND_CLIP_QUANTILE,
        "weekend_clip_value": round(weekend_clip, 4),
    }
    return train, test, split_params


def train_and_log(
    config: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    split_params: dict[str, str | float | int],
) -> str:
    """Ajusta el pipeline (scaler + K-Means) en entrenamiento, registra el run y devuelve su
    identificador."""
    features = config["features"]
    n_clusters, n_init = config["n_clusters"], config["n_init"]
    name = run_name(config)

    usage_scaler = StandardScaler().fit(train[USAGE_FEATURES])
    usage_train = usage_scaler.transform(train[USAGE_FEATURES])
    usage_test = usage_scaler.transform(test[USAGE_FEATURES])

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=n_clusters, n_init=n_init, random_state=RANDOM_STATE),
            ),
        ]
    ).fit(train[features])
    train_labels = pipeline.named_steps["kmeans"].labels_
    test_labels = pipeline.predict(test[features])

    scaler = pipeline.named_steps["scaler"]
    X_train = scaler.transform(train[features])
    X_test = scaler.transform(test[features])
    test_refit_labels = fit_kmeans(X_test, n_clusters, n_init).labels_

    metrics = {
        "train_silhouette": silhouette_score(usage_train, train_labels),
        "train_stability_ari": stability_ari(X_train, n_clusters, n_init),
        "train_min_cluster_size": int(np.bincount(train_labels).min()),
        "train_n_stations": len(train),
        "test_silhouette": silhouette_score(usage_test, test_labels),
        "test_ari_vs_refit": adjusted_rand_score(test_labels, test_refit_labels),
        "test_n_stations": len(test),
    }

    configure_mlflow()
    with mlflow.start_run(run_name=name) as active:
        log_traceability(
            "P3",
            model_family="kmeans",
            stage_of_work=config["stage_of_work"],
        )
        log_training_data(
            train[features],
            name="p3_station_profiles_train",
            source=PROCESSED_DATA_DIR.resolve().as_uri(),
        )
        mlflow.log_params(
            {
                "model": config["model"],
                "estimator_class": "KMeans",
                "n_clusters": n_clusters,
                "n_init": n_init,
                "random_state": RANDOM_STATE,
                "scaler": "StandardScaler",
                "features": ",".join(features),
                "evaluation_features": ",".join(USAGE_FEATURES),
                **split_params,
            }
        )
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        log_run_config(
            {
                **config,
                "random_state": RANDOM_STATE,
                "evaluation_features": USAGE_FEATURES,
                "n_stability_seeds": N_STABILITY_SEEDS,
                **split_params,
            }
        )

        summary = cluster_summary(train, train_labels)
        mlflow.log_text(summary.to_csv(), "data/cluster_summary.csv")
        mlflow.log_text(train.assign(cluster=train_labels).to_csv(), "data/station_clusters.csv")
        mlflow.log_text(
            test.assign(cluster=test_labels).to_csv(), "data/station_clusters_test.csv"
        )
        mlflow.log_figure(plot_clusters(train, train_labels, name), "plots/clusters.png")
        mlflow.log_figure(
            plot_clusters(test, test_labels, f"{name} (prueba)"),
            "plots/clusters_test.png",
        )

        input_example = train[features].head(5)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="modelo_p3",
            signature=infer_signature(input_example, pipeline.predict(input_example)),
            input_example=input_example,
        )

    logger.success(
        f"{name} ({active.info.run_id}): train_silhouette={metrics['train_silhouette']:.3f}, "
        f"test_ari_vs_refit={metrics['test_ari_vs_refit']:.3f}"
    )
    return active.info.run_id


def run(model: str, params: dict[str, str]) -> str:
    """Entrena un modelo de P3 en una corrida de MLflow y devuelve su identificador."""
    config = resolve_config(model, params)
    train, test, split_params = load_profiles()
    return train_and_log(config, train, test, split_params)


def main() -> None:
    """Registra el baseline y el modelo elegido con los valores por defecto."""
    train, test, split_params = load_profiles()
    for model in MODELS:
        train_and_log(resolve_config(model, {}), train, test, split_params)


if __name__ == "__main__":
    main()
