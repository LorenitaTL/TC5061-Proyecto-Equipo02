"""Punto de entrada único de entrenamiento del MLflow Project.

``mlflow run . -e train -P question=p3 -P model=kmeans_perfil -P n_clusters=4`` llega aquí como
``--question p3 --model kmeans_perfil --n_clusters 4``. MLflow agrega como ``--clave valor`` los
parámetros que no están declarados en el ``MLproject``; este módulo los reúne en ``params`` y los
entrega al adaptador de la pregunta, que convierte tipos y aplica sus valores por defecto.
"""

from importlib import import_module

from loguru import logger
import typer

ADAPTERS = {
    "p1": "mi_bici.modeling.train_p1",
    "p2": "mi_bici.modeling.train_p2",
    "p3": "mi_bici.modeling.train_p3",
}

app = typer.Typer()


def parse_extra_args(args: list[str]) -> dict[str, str]:
    """Convierte ``["--clave", "valor", ...]`` en ``{"clave": "valor"}``.

    Acepta también ``--clave=valor``. Los guiones del nombre se cambian por guiones bajos.
    """
    params: dict[str, str] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            raise typer.BadParameter(f"Argumento inesperado: {token}")
        key, sep, value = token[2:].partition("=")
        if not sep:
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                raise typer.BadParameter(f"Falta el valor de --{key}")
            value = args[i + 1]
            i += 1
        params[key.replace("-", "_")] = value
        i += 1
    return params


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    question: str = typer.Option(..., help="Pregunta a entrenar: p1, p2 o p3."),
    model: str = typer.Option("default", help="Modelo; cada adaptador define sus valores."),
) -> None:
    """Entrena un modelo de la pregunta indicada en una sola corrida de MLflow."""
    question = question.lower()
    if question not in ADAPTERS:
        raise typer.BadParameter(
            f"Pregunta desconocida: {question}. Usa p1, p2 o p3.",
            param_hint="--question",
        )

    params = parse_extra_args(ctx.args)
    module = import_module(ADAPTERS[question])
    run = getattr(module, "run", None)
    if run is None:
        raise typer.BadParameter(
            f"{ADAPTERS[question]} todavía no implementa run(model, params).",
            param_hint="--question",
        )

    run_id = run(model=model, params=params)
    logger.success(f"Corrida {run_id} de {question.upper()} terminada")


if __name__ == "__main__":
    app()
