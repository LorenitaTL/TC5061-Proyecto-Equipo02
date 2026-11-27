from types import SimpleNamespace

import pytest

from mi_bici.modeling.promote_p2 import check_version, promote

NAME = "mibici-equipo02-p2-riesgo"


def make_version(number="2", **tags):
    base = {"question": "P2", "git_commit": "abc123", "data_version": "def456"}
    base.update(tags)
    return SimpleNamespace(version=number, tags={k: v for k, v in base.items() if v is not None})


class FakeClient:
    def __init__(self, versions, aliases=None):
        self.versions = versions
        self.aliases = dict(aliases or {})

    def get_model_version(self, name, version):
        return self.versions[version]

    def get_model_version_by_alias(self, name, alias):
        if alias not in self.aliases:
            raise KeyError(alias)
        return self.versions[self.aliases[alias]]

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version


def test_check_version_accepts_a_version_with_lineage():
    assert check_version(make_version()) == []


@pytest.mark.parametrize(
    "tags, fragment",
    [
        ({"origen": "seed-sintetico"}, "semilla"),
        ({"git_commit": None}, "git_commit"),
        ({"data_version": None}, "data_version"),
        ({"question": "P1"}, "question"),
    ],
)
def test_check_version_rejects_seed_missing_lineage_or_other_question(tags, fragment):
    problems = check_version(make_version(**tags))
    assert any(fragment in p for p in problems)


def test_promote_moves_alias_and_returns_previous_version():
    client = FakeClient({"1": make_version("1"), "2": make_version("2")}, {"champion": "1"})
    assert promote(client, "2") == "1"
    assert client.aliases["champion"] == "2"


def test_revert_moves_alias_back():
    client = FakeClient({"1": make_version("1"), "2": make_version("2")}, {"champion": "2"})
    promote(client, "1")
    assert client.aliases["champion"] == "1"


def test_promote_without_existing_alias_returns_none():
    client = FakeClient({"1": make_version("1")})
    assert promote(client, "1") is None
    assert client.aliases["champion"] == "1"


def test_dry_run_does_not_move_alias():
    client = FakeClient({"1": make_version("1"), "2": make_version("2")}, {"champion": "1"})
    assert promote(client, "2", dry_run=True) == "1"
    assert client.aliases["champion"] == "1"


def test_promote_refuses_seed_model_and_keeps_alias():
    client = FakeClient(
        {"1": make_version("1"), "2": make_version("2", origen="seed-sintetico")},
        {"champion": "1"},
    )
    with pytest.raises(ValueError, match="semilla"):
        promote(client, "2")
    assert client.aliases["champion"] == "1"
