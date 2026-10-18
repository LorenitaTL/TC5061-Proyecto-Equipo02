"""Entrenamiento y registro en MLflow de P3: clustering de estaciones por perfil de uso.

Se registran dos runs en el experimento del proyecto:

- ``p3-baseline-kmeans-ubicacion-k3``: K-Means sobre latitud y longitud. Responde si basta
  la ubicación para separar estaciones con patrones de uso distintos.
- ``p3-kmeans-perfil-uso-k3``: K-Means sobre los balances de flujo en horas pico y el uso
  en fin de semana (modelo elegido en ``notebooks/1.0-lt-perfiles-estaciones.ipynb``).

Los dos modelos se ajustan con el periodo de entrenamiento (DATA_START_MONTH a
TRAIN_END_MONTH) y se evalúan en el espacio de perfiles de uso, para que sus métricas sean
comparables. Los meses posteriores se usan solo como prueba: se asignan con el modelo ya
ajustado y se compara contra un reajuste para medir si los grupos se sostienen en el tiempo.
"""

from loguru import logger
from matplotlib.figure import Figure
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import typer

from mi_bici.config import (
    DATA_END_MONTH,
    DATA_START_MONTH,
    RANDOM_STATE,
    TRAIN_END_MONTH,
)
from mi_bici.features import month_range
from mi_bici.features_p3 import (
    LOCATION_FEATURES,
    MIN_DEPARTURES,
    build_station_profiles,
    load_profile_trips,
    load_station_coordinates,
)
from mi_bici.modeling.tracking import configure_mlflow, log_traceability

app = typer.Typer()

N_CLUSTERS = 3
N_INIT = 10
N_STABILITY_SEEDS = 5
WEEKEND_CLIP_QUANTILE = 0.99

USAGE_FEATURES = ["balance_am", "balance_pm", "share_fin_semana"]

RUNS = {
    "p3-baseline-kmeans-ubicacion-k3": LOCATION_FEATURES,
    "p3-kmeans-perfil-uso-k3": USAGE_FEATURES,
}


def fit_kmeans(X: np.ndarray, seed: int = RANDOM_STATE) -> KMeans:
    """Ajusta K-Means con el número de clusters y reinicios del proyecto."""
    return KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT, random_state=seed).fit(X)


def stability_ari(X: np.ndarray, seed: int = RANDOM_STATE) -> float:
    """ARI promedio entre el ajuste con ``seed`` y ajustes con otras semillas.

    Vale 1 cuando los grupos no dependen de la inicialización. El número de cada cluster sí
    puede cambiar entre semillas; el ARI no se ve afectado por eso.
    """
    reference = fit_kmeans(X, seed).labels_
    return float(
        np.mean(
            [
                adjusted_rand_score(reference, fit_kmeans(X, seed + i).labels_)
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


def train_and_log(
    run_name: str,
    features: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    split_params: dict[str, str | float | int],
) -> None:
    """Ajusta K-Means con ``features`` en el periodo de entrenamiento y registra el run."""
    # Espacio común de evaluación: el perfil de uso, escalado solo con entrenamiento.
    usage_scaler = StandardScaler().fit(train[USAGE_FEATURES])
    usage_train = usage_scaler.transform(train[USAGE_FEATURES])
    usage_test = usage_scaler.transform(test[USAGE_FEATURES])

    scaler = StandardScaler().fit(train[features])
    X_train = scaler.transform(train[features])
    X_test = scaler.transform(test[features])

    model = fit_kmeans(X_train)
    train_labels = model.labels_
    test_labels = model.predict(X_test)
    test_refit_labels = fit_kmeans(X_test).labels_

    metrics = {
        "train_silhouette": silhouette_score(usage_train, train_labels),
        "train_stability_ari": stability_ari(X_train),
        "train_min_cluster_size": int(np.bincount(train_labels).min()),
        "train_n_stations": len(train),
        "test_silhouette": silhouette_score(usage_test, test_labels),
        "test_ari_vs_refit": adjusted_rand_score(test_labels, test_refit_labels),
        "test_n_stations": len(test),
    }

    with mlflow.start_run(run_name=run_name):
        log_traceability("P3")
        mlflow.log_params(
            {
                "model": "KMeans",
                "n_clusters": N_CLUSTERS,
                "n_init": N_INIT,
                "random_state": RANDOM_STATE,
                "scaler": "StandardScaler",
                "features": ",".join(features),
                "evaluation_features": ",".join(USAGE_FEATURES),
                **split_params,
            }
        )
        mlflow.log_metrics({name: float(value) for name, value in metrics.items()})

        assignments = train.assign(cluster=train_labels)
        mlflow.log_text(cluster_summary(train, train_labels).to_csv(), "cluster_summary.csv")
        mlflow.log_text(assignments.to_csv(), "station_clusters.csv")
        mlflow.log_figure(plot_clusters(train, train_labels, run_name), "clusters.png")

    logger.success(
        f"{run_name}: train_silhouette={metrics['train_silhouette']:.3f}, "
        f"test_ari_vs_refit={metrics['test_ari_vs_refit']:.3f}"
    )


@app.command()
def main() -> None:
    """Construye los perfiles de entrenamiento y prueba y registra los runs de P3."""
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

    experiment = configure_mlflow()
    logger.info(f"Registrando runs de P3 en el experimento {experiment}")

    for run_name, features in RUNS.items():
        train_and_log(run_name, features, train, test, split_params)


if __name__ == "__main__":
    app()
