"""Reporte de data drift y calidad de datos de P2 con Evidently.

Compara el conjunto de referencia (periodo de entrenamiento del modelo A,
enero de 2025 a marzo de 2026) contra el conjunto de evaluación A/B de junio
de 2026 que genera ``mi_bici.monitoring.prepare_p2``. Ambos conjuntos usan las
30 variables de entrada del modelo y la etiqueta ``Risk`` con el mismo umbral
calculado en entrenamiento.

Salidas en ``reports/monitoring_p2/``:

- ``p2_drift_calidad.html``: reporte interactivo de Evidently (drift + calidad).
- ``p2_drift_por_variable.csv``: método, puntaje y decisión de drift por variable.
- ``p2_drift_resumen.json``: cifras de la corrida (tamaños, semilla, drift, Risk).
- ``p2_drift_por_variable.png`` y ``p2_distribuciones_drift.png``: gráficas.
"""

import json
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from mi_bici.config import (
    P2_TRAIN_END,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from mi_bici.modeling.train_p2 import ablation_features, create_risk_label
from mi_bici.monitoring.prepare_p2 import KEY, RISK_THRESHOLD, TARGET

app = typer.Typer()

FEATURES_PATH = PROCESSED_DATA_DIR / "features_p2.parquet"
EVALUATION_PATH = PROCESSED_DATA_DIR / "monitoring_p2" / "evaluation.parquet"
OUTPUT_DIR = REPORTS_DIR / "monitoring_p2"

REFERENCE_START = "2025-01-01"
SAMPLE_SIZE = 50_000
LABEL = "Risk"
# Variables discretas con pocos valores: Evidently las compara con Jensen-Shannon.
CATEGORICAL_FEATURES = ["hour_of_day", "day_of_week", "is_weekend"]
# Métodos fijos: Evidently elige K-S/chi-cuadrada (p-valores) con <= 1,000 filas y
# distancias con más filas; fijarlos hace el resultado independiente de la muestra.
# Con distancias, hay drift cuando el puntaje >= umbral (0.1 es el valor por defecto).
NUM_METHOD = "wasserstein"
CAT_METHOD = "jensenshannon"
DRIFT_THRESHOLD = 0.1

REFERENCE_COLOR = "#2a78d6"
CURRENT_COLOR = "#eb6834"
MUTED_COLOR = "#b4b2a9"
TEXT_COLOR = "#52514e"


# ---------------------------------------------------------
# Datos
# ---------------------------------------------------------


def load_reference(features_path: Path, threshold: float = RISK_THRESHOLD) -> pd.DataFrame:
    """Construye la referencia con el periodo de entrenamiento del modelo A.

    Usa las mismas columnas y la misma regla de etiqueta que el conjunto de
    evaluación, para que cualquier diferencia provenga de los datos y no del
    procesamiento.
    """
    features = ablation_features("completo")
    reference = pd.read_parquet(
        features_path,
        columns=[*KEY, *features, TARGET],
        engine="pyarrow",
        filters=[
            ("hour", ">=", pd.Timestamp(REFERENCE_START)),
            ("hour", "<", pd.Timestamp(P2_TRAIN_END)),
        ],
    )
    reference = reference.dropna(subset=[TARGET])
    if reference.empty:
        raise ValueError("El periodo de referencia no contiene observaciones con objetivo.")

    risk = create_risk_label(reference, threshold).astype("int8")
    return reference[[*KEY, *features]].assign(**{LABEL: risk})


def sample_rows(df: pd.DataFrame, size: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Toma una muestra aleatoria reproducible; si el conjunto es menor, lo usa completo."""
    if size <= 0:
        raise ValueError("El tamaño de muestra debe ser positivo.")
    if len(df) <= size:
        return df.reset_index(drop=True)
    return df.sample(n=size, random_state=seed).reset_index(drop=True)


def to_model_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Deja solo variables del modelo y etiqueta, con tipos NumPy que Evidently acepta.

    Las columnas nullable de pandas (``Float64``, ``Int64``) se convierten a
    ``float64`` conservando los faltantes como ``NaN``. Las categóricas y la
    etiqueta quedan como ``int64`` porque Evidently no acepta etiquetas ``float``.
    """
    missing = [column for column in [*features, LABEL] if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el conjunto: {missing}")
    frame = df[[*features, LABEL]].copy()
    for column in features:
        values = pd.to_numeric(frame[column], errors="coerce")
        if column in CATEGORICAL_FEATURES:
            if values.isna().any():
                raise ValueError(f"La variable categórica {column} tiene faltantes.")
            frame[column] = values.astype("int64")
        else:
            frame[column] = values.astype("float64")
    frame[LABEL] = frame[LABEL].astype("int64")
    return frame


# ---------------------------------------------------------
# Evidently
# ---------------------------------------------------------


def build_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]):
    """Ejecuta el reporte de Evidently con drift por variable y resumen de calidad."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    categorical = [column for column in CATEGORICAL_FEATURES if column in features]
    numerical = [column for column in features if column not in categorical]
    definition = DataDefinition(
        numerical_columns=numerical,
        categorical_columns=[*categorical, LABEL],
    )
    drift = DataDriftPreset(
        num_method=NUM_METHOD,
        cat_method=CAT_METHOD,
        threshold=DRIFT_THRESHOLD,
    )
    report = Report([drift, DataSummaryPreset()], include_tests=True)
    return report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )


def drift_table(snapshot) -> pd.DataFrame:
    """Extrae del snapshot el método, el puntaje y la decisión de drift por variable."""
    rows = []
    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]
        if not config.get("type", "").endswith(":ValueDrift"):
            continue
        rows.append(
            {
                "variable": config["column"],
                "metodo": config["method"],
                "umbral": float(config["threshold"]),
                "puntaje": float(metric["value"]),
            }
        )
    if not rows:
        raise RuntimeError("El reporte de Evidently no produjo métricas de drift por variable.")
    table = pd.DataFrame(rows)
    p_value_methods = table["metodo"].str.contains("p_value", case=False)
    if p_value_methods.any():
        raise RuntimeError("Se esperaban distancias, no p-valores; revisa NUM_METHOD/CAT_METHOD.")
    table["drift"] = table["puntaje"] >= table["umbral"]
    return table.sort_values("puntaje", ascending=False).reset_index(drop=True)


def quality_summary(df: pd.DataFrame, features: list[str]) -> dict:
    """Cifras de calidad del conjunto completo, no de la muestra."""
    return {
        "filas": len(df),
        "estaciones": int(df["station_id"].nunique()),
        "claves_duplicadas": int(df.duplicated(KEY).sum()),
        "celdas_faltantes": int(df[features].isna().sum().sum()),
        "variables_con_faltantes": sorted(
            column for column in features if df[column].isna().any()
        ),
        "tasa_risk": float(df[LABEL].mean()),
    }


# ---------------------------------------------------------
# Gráficas
# ---------------------------------------------------------


def _style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)


def plot_drift_scores(table: pd.DataFrame, output: Path) -> None:
    """Barras horizontales del puntaje de drift por variable con el umbral marcado."""
    data = table.sort_values("puntaje")
    fig, ax = plt.subplots(figsize=(9, 9.5))
    colors = [CURRENT_COLOR if drift else REFERENCE_COLOR for drift in data["drift"]]
    ax.barh(data["variable"], data["puntaje"], color=colors, height=0.7)
    ax.axvline(DRIFT_THRESHOLD, color=TEXT_COLOR, linestyle="--", linewidth=1)
    ax.text(
        DRIFT_THRESHOLD,
        len(data) - 0.4,
        f" umbral {DRIFT_THRESHOLD}",
        color=TEXT_COLOR,
        fontsize=9,
        va="bottom",
    )
    for y, (score, drift) in enumerate(zip(data["puntaje"], data["drift"])):
        if drift or y >= len(data) - 5:
            ax.text(score, y, f" {score:.3f}", va="center", fontsize=8, color=TEXT_COLOR)

    predictors = table[table["variable"] != LABEL]
    label_drift = bool(table.loc[table["variable"] == LABEL, "drift"].iloc[0])
    ax.set_title(
        "P2 · drift por variable: junio 2026 vs. entrenamiento\n"
        f"{int(predictors['drift'].sum())}/{len(predictors)} predictores con drift · "
        f"Risk {'con' if label_drift else 'sin'} drift",
        loc="left",
        fontsize=11,
        color="#0b0b0b",
    )
    ax.set_xlabel(
        "Puntaje (Wasserstein normalizada en numéricas, Jensen-Shannon en categóricas)",
        color=TEXT_COLOR,
        fontsize=9,
    )
    ax.set_xlim(0, max(DRIFT_THRESHOLD, data["puntaje"].max()) * 1.15)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    _style_axis(ax)
    legend = [(CURRENT_COLOR, "Con drift"), (REFERENCE_COLOR, "Sin drift")]
    present = [(c, t) for c, t in legend if c in colors]
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in present],
        [t for _, t in present],
        frameon=False,
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _shares(values: pd.Series, bins: np.ndarray | None) -> pd.Series:
    """Proporción de filas por valor (discretas) o por intervalo (continuas)."""
    if bins is None:
        # Redondeo: senos y cosenos simétricos difieren solo por error de punto flotante.
        return values.round(6).value_counts(normalize=True).sort_index()
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.Series(counts / max(len(values), 1), index=centers)


def plot_distributions(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    table: pd.DataFrame,
    risk_rates: tuple[float, float],
    output: Path,
    n_variables: int = 3,
    n_bins: int = 40,
) -> None:
    """Compara la etiqueta Risk y las variables con mayor puntaje de drift.

    ``risk_rates`` son las tasas de los conjuntos completos, no de la muestra.
    Cada panel tiene su propio eje porque las variables viven en escalas distintas.
    Las variables con pocos valores se comparan por valor; el resto, con
    intervalos comunes a ambos conjuntos dentro de los percentiles 1-99.
    """
    top = [v for v in table["variable"] if v != LABEL][:n_variables]
    panels = [LABEL, *top]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.8))
    for ax, column in zip(np.atleast_1d(axes), panels):
        if column == LABEL:
            ax.bar(["Entrenamiento", "Junio"], risk_rates, color=[REFERENCE_COLOR, CURRENT_COLOR])
            for x, share in enumerate(risk_rates):
                ax.text(x, share, f"{share:.1%}", ha="center", va="bottom", fontsize=9)
            ax.set_title("Proporción Risk = 1 (conjunto completo)", loc="left", fontsize=10)
        else:
            ref_values = reference[column].dropna()
            cur_values = current[column].dropna()
            both = pd.concat([ref_values, cur_values])
            bins = None
            if both.nunique() > 24:
                low, high = np.percentile(both, [1, 99])
                bins = np.linspace(low, high, n_bins + 1)
                ref_values = ref_values[ref_values.between(low, high)]
                cur_values = cur_values[cur_values.between(low, high)]
            for values, color, name in (
                (ref_values, REFERENCE_COLOR, "Entrenamiento"),
                (cur_values, CURRENT_COLOR, "Junio"),
            ):
                shares = _shares(values, bins)
                ax.plot(
                    shares.index,
                    shares.values,
                    color=color,
                    linewidth=2,
                    label=name,
                    marker="o" if bins is None else None,
                    markersize=4,
                )
            score = table.loc[table["variable"] == column, "puntaje"].iloc[0]
            ax.set_title(f"{column} (puntaje {score:.3f})", loc="left", fontsize=10)
            ax.set_ylabel("Proporción de filas", fontsize=9, color=TEXT_COLOR)
            ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        _style_axis(ax)
    fig.suptitle(
        "P2 · etiqueta y las tres variables con mayor puntaje de drift",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------
# Comando
# ---------------------------------------------------------


@app.command()
def main(
    features_path: Path = FEATURES_PATH,
    evaluation_path: Path = EVALUATION_PATH,
    output_dir: Path = OUTPUT_DIR,
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_STATE,
) -> None:
    """Genera el reporte de drift y calidad de P2 (entrenamiento vs. junio de 2026).

    ``features_path`` es el features_p2.parquet completo; ``evaluation_path``, la
    salida de prepare_p2; ``sample_size``, las filas por conjunto que recibe
    Evidently, y ``seed``, la semilla del muestreo.
    """
    for path in (features_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró {path}. Revisa la sección P2 del README.")

    features = ablation_features("completo")
    logger.info("Cargando referencia (entrenamiento) y conjunto actual (junio)")
    reference_full = load_reference(features_path)
    current_full = pd.read_parquet(evaluation_path)

    reference = to_model_frame(sample_rows(reference_full, sample_size, seed), features)
    current = to_model_frame(sample_rows(current_full, sample_size, seed), features)
    logger.info(f"Muestras: referencia {len(reference):,} | actual {len(current):,}")

    snapshot = build_report(reference, current, features)
    table = drift_table(snapshot)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "p2_drift_calidad.html"
    snapshot.save_html(str(html_path))
    table.to_csv(output_dir / "p2_drift_por_variable.csv", index=False)
    plot_drift_scores(table, output_dir / "p2_drift_por_variable.png")
    risk_rates = (float(reference_full[LABEL].mean()), float(current_full[LABEL].mean()))
    plot_distributions(
        reference, current, table, risk_rates, output_dir / "p2_distribuciones_drift.png"
    )

    import evidently

    predictors = table[table["variable"] != LABEL]
    summary = {
        "evidently": evidently.__version__,
        "referencia": f"{REFERENCE_START} a {P2_TRAIN_END} (exclusivo)",
        "actual": "2026-06-01 a 2026-07-01 (exclusivo)",
        "umbral_risk": RISK_THRESHOLD,
        "muestra_por_conjunto": sample_size,
        "semilla": seed,
        "umbral_drift": DRIFT_THRESHOLD,
        "predictores_con_drift": predictors.loc[predictors["drift"], "variable"].tolist(),
        "share_predictores_con_drift": float(predictors["drift"].mean()),
        "drift_en_risk": bool(table.loc[table["variable"] == LABEL, "drift"].iloc[0]),
        "calidad_referencia": quality_summary(reference_full, features),
        "calidad_actual": quality_summary(current_full, features),
    }
    (output_dir / "p2_drift_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.success(f"Reporte: {html_path}")
    logger.info(
        f"Predictores con drift: {len(summary['predictores_con_drift'])}/{len(predictors)} "
        f"| drift en Risk: {summary['drift_en_risk']}"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    app()
