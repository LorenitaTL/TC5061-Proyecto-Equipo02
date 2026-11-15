"""Compara dos versiones registradas del modelo P2 sobre el mismo conjunto de evaluación.

Uso::

    python -m mi_bici.monitoring.ab_test_p2 --model-a champion --model-b challenger

``--model-a`` y ``--model-b`` aceptan un alias (``champion``) o un número de versión (``5``).
Las dos versiones se evalúan sobre las mismas filas de ``evaluation.parquet`` (junio de 2026,
generado con ``prepare_p2``) y con el mismo umbral de clasificación, así que la única
diferencia entre ellas es el modelo.

Además de las métricas puntuales se reporta:

- la prueba de McNemar sobre los aciertos de cada versión fila por fila;
- intervalos de confianza al 95 % de la diferencia B − A, con bootstrap por estación (las
  horas de una misma estación no son independientes, así que se remuestrean estaciones);
- las métricas de test que cada versión registró al entrenarse, como control: junio es el
  periodo de test de P2, así que deben coincidir con las que se calculan aquí.

Regla de decisión, fijada antes de ver los resultados: Recall es la métrica principal de P2.
B reemplaza a A solo si su Recall es significativamente mayor (el IC 95 % de ΔRecall queda
por encima de 0) y su F1 no es significativamente menor (el IC 95 % de ΔF1 no queda por
debajo de 0). En cualquier otro caso se conserva la versión vigente.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Annotated

import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import typer

from mi_bici.config import (
    P2_CLASSIFICATION_THRESHOLD,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from mi_bici.modeling.register_p2 import MODEL_NAME
from mi_bici.modeling.tracking import configure_mlflow, log_traceability
from mi_bici.modeling.train_p2 import ablation_features

# Salida por defecto de ``python -m mi_bici.monitoring.prepare_p2``.
EVALUATION_PATH = PROCESSED_DATA_DIR / "monitoring_p2" / "evaluation.parquet"
OUTPUT_DIR = REPORTS_DIR / "ab_test_p2"
TARGET = "Risk"
GROUP = "station_id"
METRICS = ("recall", "precision", "f1", "accuracy", "roc_auc", "pr_auc")
# Métricas cuya diferencia lleva intervalo de confianza (se calculan con conteos por estación).
BOOTSTRAP_METRICS = ("recall", "precision", "f1")
N_BOOTSTRAP = 2000

app = typer.Typer()


@dataclass(frozen=True)
class ModelRef:
    """Versión del Registry resuelta a partir de un alias o un número de versión."""

    label: str
    reference: str
    version: str
    run_id: str


# ---------------------------------------------------------
# Métricas
# ---------------------------------------------------------


def predict(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Convierte probabilidades en la clase Risk con el mismo criterio que ``train_p2``."""
    return (probabilities >= threshold).astype(int)


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calcula las métricas de clasificación de P2 para una versión del modelo."""
    predictions = predict(probabilities, threshold)
    return {
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def mcnemar_test(
    y_true: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
) -> dict[str, float]:
    """Prueba exacta de McNemar sobre las filas en que solo una versión acierta.

    ``only_a_correct`` y ``only_b_correct`` son las celdas discordantes; si las dos versiones
    aciertan y fallan en las mismas filas, no hay evidencia de diferencia (p = 1).
    """
    correct_a = predictions_a == y_true
    correct_b = predictions_b == y_true
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    discordant = only_a + only_b
    p_value = 1.0 if discordant == 0 else binomtest(only_a, discordant, 0.5).pvalue
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "p_value": float(p_value),
    }


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide sin advertencias y devuelve 0 donde el denominador es 0 (como ``zero_division``)."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _metrics_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> dict:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "recall": recall,
        "precision": precision,
        "f1": _ratio(2 * precision * recall, precision + recall),
    }


def bootstrap_deltas(
    groups: np.ndarray,
    y_true: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> dict[str, tuple[float, float]]:
    """IC 95 % de la diferencia B − A en Recall, Precision y F1, remuestreando estaciones.

    Se cuentan TP, FP y FN por estación y versión; cada réplica del bootstrap toma estaciones
    con reemplazo y suma sus conteos, de modo que las dos versiones se comparan siempre sobre
    la misma muestra (comparación pareada).
    """
    frame = pd.DataFrame(
        {
            "group": groups,
            "tp_a": (predictions_a == 1) & (y_true == 1),
            "fp_a": (predictions_a == 1) & (y_true == 0),
            "fn_a": (predictions_a == 0) & (y_true == 1),
            "tp_b": (predictions_b == 1) & (y_true == 1),
            "fp_b": (predictions_b == 1) & (y_true == 0),
            "fn_b": (predictions_b == 0) & (y_true == 1),
        }
    )
    counts = frame.groupby("group").sum()
    n_groups = len(counts)

    rng = np.random.default_rng(seed)
    # Cuántas veces entra cada estación en cada réplica.
    weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_bootstrap)
    totals = weights @ counts.to_numpy(dtype=float)
    columns = {name: totals[:, i] for i, name in enumerate(counts.columns)}

    metrics_a = _metrics_from_counts(columns["tp_a"], columns["fp_a"], columns["fn_a"])
    metrics_b = _metrics_from_counts(columns["tp_b"], columns["fp_b"], columns["fn_b"])

    intervals = {}
    for name in BOOTSTRAP_METRICS:
        delta = metrics_b[name] - metrics_a[name]
        low, high = np.percentile(delta, [2.5, 97.5])
        intervals[name] = (float(low), float(high))
    return intervals


def decide(intervals: dict[str, tuple[float, float]]) -> tuple[str, str]:
    """Aplica la regla de decisión del módulo y devuelve la versión ganadora y el motivo."""
    recall_low, recall_high = intervals["recall"]
    _, f1_high = intervals["f1"]

    if recall_low > 0 and f1_high >= 0:
        return (
            "B",
            "B mejora Recall de forma significativa sin empeorar F1 de forma significativa.",
        )
    if recall_low > 0:
        return "A", "B mejora Recall, pero su F1 es significativamente menor."
    if recall_high < 0:
        return "A", "B tiene un Recall significativamente menor que A."
    return "A", "No hay diferencia significativa en Recall; se conserva la versión vigente."


# ---------------------------------------------------------
# Registry
# ---------------------------------------------------------


def resolve_model(client: MlflowClient, name: str, reference: str, label: str) -> ModelRef:
    """Resuelve un alias o número de versión a la versión concreta del Registry."""
    if reference.isdigit():
        model_version = client.get_model_version(name, reference)
    else:
        model_version = client.get_model_version_by_alias(name, reference)
    return ModelRef(
        label=label,
        reference=reference,
        version=str(model_version.version),
        run_id=model_version.run_id,
    )


def logged_test_metrics(client: MlflowClient, run_id: str) -> dict[str, float]:
    """Métricas de test que registró el run de entrenamiento de la versión."""
    metrics = client.get_run(run_id).data.metrics
    return {name: metrics[f"test_{name}"] for name in METRICS if f"test_{name}" in metrics}


# ---------------------------------------------------------
# Reporte
# ---------------------------------------------------------


def build_table(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    intervals: dict[str, tuple[float, float]],
    logged_a: dict[str, float],
    logged_b: dict[str, float],
) -> pd.DataFrame:
    """Tabla comparativa: una fila por métrica, con la diferencia B − A y su IC 95 %."""
    rows = []
    for name in METRICS:
        low, high = intervals.get(name, (np.nan, np.nan))
        rows.append(
            {
                "metrica": name,
                "A": metrics_a[name],
                "B": metrics_b[name],
                "delta_B_menos_A": metrics_b[name] - metrics_a[name],
                "ic95_bajo": low,
                "ic95_alto": high,
                "A_test_registrado": logged_a.get(name, np.nan),
                "B_test_registrado": logged_b.get(name, np.nan),
            }
        )
    return pd.DataFrame(rows)


def table_to_markdown(table: pd.DataFrame) -> str:
    """Convierte la tabla a Markdown sin depender de ``tabulate``."""

    def fmt(value) -> str:
        if isinstance(value, str):
            return value
        return "—" if pd.isna(value) else f"{value:.4f}"

    header = "| " + " | ".join(table.columns) + " |"
    separator = "|" + "|".join(["---"] + ["---:"] * (len(table.columns) - 1)) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, separator, *body]) + "\n"


def max_logged_gap(table: pd.DataFrame) -> float:
    """Mayor diferencia absoluta entre lo calculado aquí y lo registrado en test."""
    gaps = pd.concat(
        [
            (table["A"] - table["A_test_registrado"]).abs(),
            (table["B"] - table["B_test_registrado"]).abs(),
        ]
    ).dropna()
    return float(gaps.max()) if not gaps.empty else float("nan")


# ---------------------------------------------------------
# Ejecución
# ---------------------------------------------------------


def load_evaluation(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Carga el conjunto común de evaluación y valida las columnas que usa el comparador."""
    features = ablation_features("completo")
    evaluation = pd.read_parquet(path)
    missing = sorted({*features, TARGET, GROUP} - set(evaluation.columns))
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {missing}")
    return evaluation[features], evaluation[TARGET], evaluation[GROUP]


@app.command()
def main(
    model_a: str = typer.Option("champion", help="Versión A: alias o número de versión."),
    model_b: str = typer.Option("challenger", help="Versión B: alias o número de versión."),
    name: str = typer.Option(MODEL_NAME, help="Nombre del modelo en el Registry."),
    evaluation_path: Annotated[
        Path, typer.Option(help="Conjunto común de evaluación.")
    ] = EVALUATION_PATH,
    threshold: float = typer.Option(
        P2_CLASSIFICATION_THRESHOLD, help="Umbral de clasificación, igual para A y B."
    ),
    output_dir: Annotated[
        Path, typer.Option(help="Carpeta de la tabla y el resumen.")
    ] = OUTPUT_DIR,
    log_mlflow: bool = typer.Option(True, help="Registrar la comparación como run de MLflow."),
) -> None:
    """Evalúa A y B sobre las mismas filas y aplica la regla de decisión."""
    configure_mlflow()
    client = MlflowClient()

    ref_a = resolve_model(client, name, model_a, "A")
    ref_b = resolve_model(client, name, model_b, "B")
    typer.echo(f"A: {name} v{ref_a.version} ({model_a}), run {ref_a.run_id}")
    typer.echo(f"B: {name} v{ref_b.version} ({model_b}), run {ref_b.run_id}")

    X, y, groups = load_evaluation(evaluation_path)
    y_true = y.to_numpy()
    typer.echo(
        f"Evaluación: {len(X):,} filas, {groups.nunique()} estaciones, Risk = {y.mean():.2%}"
    )

    probabilities = {}
    for ref in (ref_a, ref_b):
        model = mlflow.sklearn.load_model(f"models:/{name}/{ref.version}")
        probabilities[ref.label] = model.predict_proba(X)[:, 1]

    metrics_a = compute_metrics(y_true, probabilities["A"], threshold)
    metrics_b = compute_metrics(y_true, probabilities["B"], threshold)
    predictions_a = predict(probabilities["A"], threshold)
    predictions_b = predict(probabilities["B"], threshold)

    mcnemar = mcnemar_test(y_true, predictions_a, predictions_b)
    intervals = bootstrap_deltas(groups.to_numpy(), y_true, predictions_a, predictions_b)
    winner, reason = decide(intervals)

    table = build_table(
        metrics_a,
        metrics_b,
        intervals,
        logged_test_metrics(client, ref_a.run_id),
        logged_test_metrics(client, ref_b.run_id),
    )
    gap = max_logged_gap(table)

    summary = {
        "model_name": name,
        "A": {"reference": model_a, "version": ref_a.version, "run_id": ref_a.run_id},
        "B": {"reference": model_b, "version": ref_b.version, "run_id": ref_b.run_id},
        "evaluation_path": str(evaluation_path),
        "rows": len(X),
        "stations": int(groups.nunique()),
        "risk_rate": float(y.mean()),
        "threshold": threshold,
        "n_bootstrap": N_BOOTSTRAP,
        "mcnemar": mcnemar,
        "max_gap_vs_logged_test": gap,
        "winner": winner,
        "reason": reason,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "ab_metrics.csv"
    markdown_path = output_dir / "ab_metrics.md"
    summary_path = output_dir / "ab_summary.json"
    table.to_csv(table_path, index=False)
    markdown_path.write_text(table_to_markdown(table), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    typer.echo("")
    typer.echo(table_to_markdown(table))
    typer.echo(
        f"McNemar: solo A acierta {mcnemar['only_a_correct']:,}, "
        f"solo B acierta {mcnemar['only_b_correct']:,}, p = {mcnemar['p_value']:.4g}"
    )
    typer.echo(f"Diferencia máxima contra las métricas de test registradas: {gap:.2e}")
    typer.echo(f"Decisión: {winner}. {reason}")
    typer.echo(f"Archivos: {table_path}, {markdown_path}, {summary_path}")

    if log_mlflow:
        with mlflow.start_run(run_name=f"P2-ab-test-v{ref_a.version}-vs-v{ref_b.version}"):
            log_traceability("P2")
            mlflow.set_tag("stage_of_work", "ab_test")
            mlflow.log_params(
                {
                    "model_name": name,
                    "version_a": ref_a.version,
                    "version_b": ref_b.version,
                    "classification_threshold": threshold,
                    "n_bootstrap": N_BOOTSTRAP,
                    "evaluation_rows": len(X),
                }
            )
            for metric, value in metrics_a.items():
                mlflow.log_metric(f"a_{metric}", value)
            for metric, value in metrics_b.items():
                mlflow.log_metric(f"b_{metric}", value)
            mlflow.log_metric("mcnemar_p_value", mcnemar["p_value"])
            mlflow.set_tag("ab_winner", winner)
            for path in (table_path, markdown_path, summary_path):
                mlflow.log_artifact(str(path))


if __name__ == "__main__":
    app()
