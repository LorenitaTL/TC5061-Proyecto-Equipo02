"""Mueve el alias de P2 a otra versión del Registry: promover o revertir.

Uso::

    python -m mi_bici.modeling.promote_p2 --version 2
    python -m mi_bici.modeling.promote_p2 --version 1 --alias champion   # revertir
    python -m mi_bici.modeling.promote_p2 --version 2 --dry-run

Antes de mover el alias revisa que la versión exista, que sea de P2, que tenga linaje
(``git_commit`` y ``data_version``) y que no sea el modelo semilla sintético (tag
``origen=seed-sintetico``). Después comprueba que el alias apunta a la versión pedida.

La API carga el modelo una sola vez al arrancar, así que hay que reiniciarla para que sirva
la versión nueva. Después se corre ``make champion-card-p2`` para actualizar la ficha.
"""

from __future__ import annotations

from typing import Annotated

import typer

from mi_bici.config import MLFLOW_TRACKING_URI

MODEL_NAME = "mibici-equipo02-p2-riesgo"
ALIAS = "champion"
SEED_TAG = ("origen", "seed-sintetico")
REQUIRED_TAGS = ("git_commit", "data_version")

app = typer.Typer()


def check_version(version) -> list[str]:
    """Devuelve los motivos por los que una versión no puede promoverse (vacío si puede)."""
    tags = version.tags or {}
    problems = []
    if tags.get("question") != "P2":
        problems.append(f"el tag question es {tags.get('question')!r}, no 'P2'")
    if tags.get(SEED_TAG[0]) == SEED_TAG[1]:
        problems.append("es el modelo semilla sintético; no debe ser champion")
    problems.extend(f"falta el tag de linaje {t}" for t in REQUIRED_TAGS if not tags.get(t))
    return problems


def promote(client, version: str, alias: str = ALIAS, name: str = MODEL_NAME, dry_run=False):
    """Asigna ``alias`` a ``version``. Devuelve la versión que tenía el alias antes."""
    candidate = client.get_model_version(name, version)
    problems = check_version(candidate)
    if problems:
        raise ValueError(f"La versión {version} no puede promoverse: " + "; ".join(problems))
    try:
        previous = client.get_model_version_by_alias(name, alias).version
    except Exception:  # noqa: BLE001  (el alias puede no existir todavía)
        previous = None
    if dry_run:
        return previous
    client.set_registered_model_alias(name, alias, version)
    resolved = client.get_model_version_by_alias(name, alias).version
    if str(resolved) != str(version):
        raise RuntimeError(f"El alias {alias} apunta a la versión {resolved}, no a {version}.")
    return previous


@app.command()
def main(
    version: Annotated[str, typer.Option(help="Número de versión que recibirá el alias.")],
    alias: Annotated[str, typer.Option(help="Alias a mover.")] = ALIAS,
    name: Annotated[str, typer.Option(help="Nombre del modelo en el Registry.")] = MODEL_NAME,
    dry_run: Annotated[bool, typer.Option(help="Solo valida; no mueve el alias.")] = False,
) -> None:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        previous = promote(MlflowClient(), version, alias, name, dry_run)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if dry_run:
        typer.echo(f"OK (simulacro): {alias} pasaría de v{previous} a v{version}.")
        return
    typer.echo(f"OK: {alias} pasó de v{previous} a v{version}.")
    typer.echo(f"Para revertir: make promote-p2 VERSION={previous}" if previous else "")
    typer.echo("Reinicia la API (make compose-down && make compose-up) y corre "
               "make champion-card-p2.")


if __name__ == "__main__":
    app()
