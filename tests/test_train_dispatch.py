import sys
from types import ModuleType

import pytest
import typer
from typer.testing import CliRunner

from mi_bici.modeling import train
from mi_bici.modeling.train import app, parse_extra_args

runner = CliRunner()


@pytest.fixture
def fake_adapter(monkeypatch):
    """Sustituye el adaptador de P3 por uno falso que solo guarda con qué se le llamó."""
    calls = []
    module = ModuleType("fake_p3")

    def run(model, params):
        calls.append({"model": model, "params": params})
        return "run-123"

    module.run = run
    monkeypatch.setitem(sys.modules, "fake_p3", module)
    monkeypatch.setitem(train.ADAPTERS, "p3", "fake_p3")
    return calls


def test_parse_extra_args_accepts_both_formats():
    args = ["--n_clusters", "4", "--learning-rate=0.05", "--alpha", "-1"]

    assert parse_extra_args(args) == {"n_clusters": "4", "learning_rate": "0.05", "alpha": "-1"}


def test_parse_extra_args_rejects_missing_value():
    with pytest.raises(typer.BadParameter, match="Falta el valor de --n_clusters"):
        parse_extra_args(["--n_clusters", "--seed", "1"])


def test_parse_extra_args_rejects_positional_values():
    with pytest.raises(typer.BadParameter, match="Argumento inesperado"):
        parse_extra_args(["4"])


def test_main_passes_model_and_extra_params_to_adapter(fake_adapter):
    result = runner.invoke(app, ["--question", "P3", "--model", "kmeans", "--n_clusters", "4"])

    assert result.exit_code == 0, result.output
    assert fake_adapter == [{"model": "kmeans", "params": {"n_clusters": "4"}}]


def test_main_uses_default_model(fake_adapter):
    result = runner.invoke(app, ["--question", "p3"])

    assert result.exit_code == 0, result.output
    assert fake_adapter == [{"model": "default", "params": {}}]


def test_main_rejects_unknown_question():
    result = runner.invoke(app, ["--question", "p9"])

    assert result.exit_code != 0
    assert "Pregunta desconocida" in result.output


def test_main_rejects_adapter_without_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "empty_p3", ModuleType("empty_p3"))
    monkeypatch.setitem(train.ADAPTERS, "p3", "empty_p3")

    result = runner.invoke(app, ["--question", "p3"])

    assert result.exit_code != 0
    assert "todavía no implementa" in result.output
