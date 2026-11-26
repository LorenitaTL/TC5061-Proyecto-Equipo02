"""Genera la ficha técnica del champion de P2 leyendo el Model Registry, no el README.

Uso::

    python -m mi_bici.modeling.champion_card_p2
    python -m mi_bici.modeling.champion_card_p2 --alias challenger --output -
    python -m mi_bici.modeling.champion_card_p2 --output reports/p2_champion_registry.md

La ficha reúne en un solo lugar lo que la API sirve de verdad: nombre, versión, alias, familia,
umbral de clasificación, linaje (run de origen, ``git_commit`` y ``data_version``), periodos
de los datos y métricas de test del run de origen. Además revisa que esos valores coincidan
con lo que el código espera (``mi_bici/config.py``) y avisa si la versión es el modelo semilla
sintético que registra Compose (tag ``origen=seed-sintetico``), que no debe presentarse como el
modelo entrenado con datos reales.

Termina con código 1 si alguna verificación falla, para poder usarla antes de grabar la demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import typer

from mi_bici.config import (
    MLFLOW_TRACKING_URI,
    P2_CLASSIFICATION_THRESHOLD,
    P2_HGB_LEARNING_RATE,
    P2_HGB_MAX_ITER,
    P2_HGB_MAX_LEAF_NODES,
    P2_HGB_MIN_SAMPLES_LEAF,
    PROJ_ROOT,
)

MODEL_NAME = "mibici-equipo02-p2-riesgo"
ALIAS = "champion"
EXPECTED_FAMILY = "HistGradientBoosting"
# Run de origen del champion documentado en docs/p2_ficha_champion.md. Distingue el Registry
# compartido de los registros locales, que pueden tener otro run con la misma configuración.
EXPECTED_RUN_ID = "55d8255b3272402ea2532729ec2b1629"
SEED_TAG = ("origen", "seed-sintetico")
DEFAULT_OUTPUT = PROJ_ROOT / "reports" / "p2_champion_registry.md"

# Métricas de test que train_p2 registra como test_<métrica>.
TEST_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")

# Hiperparámetros esperados del champion según la configuración del proyecto.
EXPECTED_PARAMS = {
    "classification_threshold": P2_CLASSIFICATION_THRESHOLD,
    "learning_rate": P2_HGB_LEARNING_RATE,
    "max_iter": P2_HGB_MAX_ITER,
    "max_leaf_nodes": P2_HGB_MAX_LEAF_NODES,
    "min_samples_leaf": P2_HGB_MIN_SAMPLES_LEAF,
}

app = typer.Typer()


@dataclass
class ChampionCard:
    """Valores de la ficha, tal como salen del Registry y del run de origen."""

    model_name: str
    alias: str
    version: str
    aliases: list[str]
    registered_at: str
    run_id: str
    run_name: str
    model_family: str
    classification_threshold: str
    git_commit: str
    data_version: str
    features_data_version: str
    train_end: str
    validation_end: str
    risk_percentile: str
    risk_threshold: str
    hyperparameters: dict[str, str]
    test_metrics: dict[str, float]
    is_seed: bool
    tracking_uri: str
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)


def _first(*values) -> str:
    """Primer valor no vacío, como texto."""
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def _same_number(actual: str, expected: float) -> bool:
    try:
        return abs(float(actual) - float(expected)) < 1e-9
    except (TypeError, ValueError):
        return False


def build_card(
    client,
    name: str = MODEL_NAME,
    alias: str = ALIAS,
    tracking_uri: str = "",
    expected_run_id: str = EXPECTED_RUN_ID,
) -> ChampionCard:
    """Resuelve el alias, lee la versión y su run de origen y arma la ficha con verificaciones."""

    version = client.get_model_version_by_alias(name, alias)
    tags = dict(version.tags or {})
    run = client.get_run(version.run_id)
    run_tags = run.data.tags
    params = run.data.params
    metrics = run.data.metrics

    def value(key: str) -> str:
        # El tag de la versión es la fuente oficial; el run de origen es el respaldo.
        return _first(tags.get(key), params.get(key), run_tags.get(key))

    threshold = _first(
        tags.get("classification_threshold"),
        params.get("classification_threshold"),
        metrics.get("classification_threshold"),
    )
    hyperparameters = {
        key: value(key)
        for key in ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf")
        if value(key)
    }
    test_metrics = {
        metric: float(metrics[f"test_{metric}"])
        for metric in TEST_METRICS
        if f"test_{metric}" in metrics
    }
    registered_at = ""
    if getattr(version, "creation_timestamp", None):
        registered_at = datetime.fromtimestamp(version.creation_timestamp / 1000, tz=UTC).strftime(
            "%Y-%m-%d %H:%M UTC"
        )

    aliases = list(getattr(version, "aliases", None) or [alias])

    card = ChampionCard(
        model_name=name,
        alias=alias,
        version=str(version.version),
        aliases=aliases,
        registered_at=registered_at,
        run_id=version.run_id,
        run_name=_first(tags.get("source_run_name"), run_tags.get("mlflow.runName")),
        model_family=value("model_family"),
        classification_threshold=threshold,
        git_commit=value("git_commit"),
        data_version=value("data_version"),
        features_data_version=value("features_data_version"),
        train_end=value("train_end"),
        validation_end=value("validation_end"),
        risk_percentile=value("risk_percentile"),
        risk_threshold=value("risk_threshold"),
        hyperparameters=hyperparameters,
        test_metrics=test_metrics,
        is_seed=tags.get(SEED_TAG[0]) == SEED_TAG[1],
        tracking_uri=tracking_uri,
    )
    card.checks = run_checks(card, tags, expected_run_id)
    return card


def run_checks(
    card: ChampionCard, tags: dict[str, str], expected_run_id: str = ""
) -> list[tuple[str, bool, str]]:
    """Compara la ficha contra lo que el código y la documentación esperan."""

    checks = []
    if expected_run_id:
        checks.append(
            (
                "Es el run documentado en docs/p2_ficha_champion.md",
                card.run_id == expected_run_id,
                (
                    f"versión {card.version}, run {card.run_id}; esperado {expected_run_id}. Si no "
                    "coincide, este Registry no es el compartido o el champion cambió."
                ),
            )
        )
    checks += [
        (
            "No es el modelo semilla sintético",
            not card.is_seed,
            "La versión tiene origen=seed-sintetico: sirve para operar y medir carga, no "
            "para presentarse como el mejor modelo."
            if card.is_seed
            else "Sin tag origen=seed-sintetico.",
        ),
        (
            f"Familia {EXPECTED_FAMILY}",
            card.model_family == EXPECTED_FAMILY,
            f"model_family = {card.model_family or '(vacío)'}",
        ),
        (
            "Linaje completo (git_commit y data_version)",
            bool(card.git_commit and card.data_version),
            (
                f"git_commit = {card.git_commit or '(vacío)'}; "
                f"data_version = {card.data_version or '(vacío)'}"
            ),
        ),
        (
            "El tag source_run_id coincide con el run de la versión",
            tags.get("source_run_id", card.run_id) == card.run_id,
            f"source_run_id = {tags.get('source_run_id', '(sin tag)')}; run = {card.run_id}",
        ),
    ]
    for key, expected in EXPECTED_PARAMS.items():
        actual = (
            card.classification_threshold
            if key == "classification_threshold"
            else card.hyperparameters.get(key, "")
        )
        checks.append(
            (
                f"{key} = {expected} (mi_bici/config.py)",
                _same_number(actual, expected),
                f"Registry: {actual or '(vacío)'}",
            )
        )
    return checks


def _fmt(value: str) -> str:
    return f"`{value}`" if value else "(no registrado)"


def render_markdown(card: ChampionCard) -> str:
    """Ficha en Markdown, que se guarda en reports/p2_champion_registry.md."""

    metric_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }
    lines = [
        "# Ficha técnica del champion de P2",
        "",
        (
            "> Generada con `python -m mi_bici.modeling.champion_card_p2` a partir del Model "
            "Registry. No se edita a mano: si cambia el champion, se vuelve a generar."
        ),
        "",
        "| Campo | Valor | Fuente |",
        "| :--- | :--- | :--- |",
        f"| Registered model | {_fmt(card.model_name)} | MLflow Registry |",
        f"| Alias operativo | {_fmt(card.alias)} | MLflow Registry |",
        f"| Otros aliases de la versión | {', '.join(f'`{a}`' for a in card.aliases)} | MLflow Registry |",
        f"| Versión | {_fmt(card.version)} | Model version |",
        f"| Registrada | {card.registered_at or '(sin fecha)'} | Model version |",
        f"| Familia | {_fmt(card.model_family)} | Tag `model_family` |",
        (
            f"| Umbral de clasificación | {_fmt(card.classification_threshold)} | "
            "Tag/parámetro `classification_threshold` |"
        ),
        f"| Hiperparámetros | {', '.join(f'`{k}={v}`' for k, v in card.hyperparameters.items()) or '(no registrados)'} | Tags de la versión |",
        f"| Run de origen | {_fmt(card.run_id)} ({card.run_name or 'sin nombre'}) | `source_run_id` |",
        f"| Commit | {_fmt(card.git_commit)} | Tag `git_commit` |",
        f"| Versión de datos | {_fmt(card.data_version)} | Tag `data_version` |",
        f"| Versión de features | {_fmt(card.features_data_version)} | `features_data_version` |",
        (
            f"| Ventanas | train < {card.train_end or '?'}; validation < {card.validation_end or '?'}; "
            "test = junio 2026 | `train_end`, `validation_end` |"
        ),
        (
            f"| Etiqueta Risk | P{_pct(card.risk_percentile)} de train, umbral {_fmt(card.risk_threshold)} | "
            "`risk_percentile`, `risk_threshold` |"
        ),
        f"| Origen | {'**Modelo semilla sintético** (`origen=seed-sintetico`)' if card.is_seed else 'Entrenado con datos reales de MiBici'} | Tag `origen` |",
        (
            "| Estado de producción | Versión a la que apunta `champion`. MLflow 3 usa aliases en "
            "lugar de stages (`Staging`/`Production`) | README y Registry |"
        ),
        "",
        "## Métricas de test del run de origen",
        "",
    ]
    if card.test_metrics:
        lines += [
            "| " + " | ".join(metric_names[m] for m in card.test_metrics) + " |",
            "|" + "---:|" * len(card.test_metrics),
            "| " + " | ".join(f"{v:.4f}" for v in card.test_metrics.values()) + " |",
        ]
    else:
        lines.append("El run de origen no tiene métricas `test_*`.")
    lines += [
        "",
        "## Verificaciones",
        "",
        "| Verificación | Resultado | Detalle |",
        "| :--- | :---: | :--- |",
    ]
    for name, passed, detail in card.checks:
        lines.append(f"| {name} | {'OK' if passed else 'FALLA'} | {detail} |")
    lines += [
        "",
        f"Resultado: **{'todas las verificaciones pasan' if card.ok else 'hay verificaciones con FALLA'}**.",
        "",
    ]
    return "\n".join(lines)


def _pct(value: str) -> str:
    try:
        return f"{float(value) * 100:.0f}"
    except (TypeError, ValueError):
        return "?"


@app.command()
def main(
    name: str = typer.Option(MODEL_NAME, help="Modelo registrado."),
    alias: str = typer.Option(ALIAS, help="Alias a resolver."),
    expected_run_id: str = typer.Option(
        EXPECTED_RUN_ID,
        help="Run de origen esperado; '' desactiva la verificación (por ejemplo, para el seed).",
    ),
    output: str = typer.Option(
        str(DEFAULT_OUTPUT), help="Archivo Markdown de salida; '-' imprime en pantalla."
    ),
) -> None:
    """Lee el Registry, escribe la ficha y falla si alguna verificación no pasa."""

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    card = build_card(
        MlflowClient(),
        name=name,
        alias=alias,
        tracking_uri=MLFLOW_TRACKING_URI,
        expected_run_id=expected_run_id,
    )
    text = render_markdown(card)

    if output == "-":
        typer.echo(text)
    else:
        Path(output).write_text(text, encoding="utf-8")
        typer.echo(f"Ficha escrita en {output}")

    typer.echo(f"Registry: {MLFLOW_TRACKING_URI}")
    typer.echo(f"{name}@{alias} -> versión {card.version}, run {card.run_id}")
    for check, passed, detail in card.checks:
        typer.echo(f"[{'OK' if passed else 'FALLA'}] {check}: {detail}")

    if not card.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
