from types import SimpleNamespace

from mi_bici.modeling.champion_card_p2 import build_card, render_markdown

RUN_ID = "55d8255b3272402ea2532729ec2b1629"


def make_version(**extra_tags):
    tags = {
        "question": "P2",
        "source_run_id": RUN_ID,
        "source_run_name": "P2-HistGradientBoosting-candidate",
        "git_commit": "cb67db696e9316e41e4dc20929f",  # prefijo visible en la captura
        "data_version": "198acbc4c315b252808c926b4",  # prefijo visible en la captura
        "model_family": "HistGradientBoosting",
        "train_end": "2026-04-01",
        "validation_end": "2026-06-01",
        "risk_percentile": "0.8",
        "risk_threshold": "0.06666666666666667",
        "classification_threshold": "0.35",
        "learning_rate": "0.1",
        "max_iter": "200",
        "max_leaf_nodes": "31",
        "min_samples_leaf": "20",
    }
    tags.update(extra_tags)
    return SimpleNamespace(
        version="1",
        run_id=RUN_ID,
        tags={k: v for k, v in tags.items() if v is not None},
        aliases=["champion"],
        creation_timestamp=1793587204000,  # 01/11/2026 20:40:04 en UTC-6, como la captura
    )


class FakeClient:
    def __init__(self, version):
        self.version = version

    def get_model_version_by_alias(self, name, alias):
        assert alias == "champion"
        return self.version

    def get_run(self, run_id):
        return SimpleNamespace(
            data=SimpleNamespace(
                tags={"mlflow.runName": "P2-HistGradientBoosting-candidate"},
                params={},
                metrics={"test_recall": 0.6871, "test_f1": 0.5719, "test_roc_auc": 0.8786},
            )
        )


def test_real_champion_passes_all_checks():
    card = build_card(FakeClient(make_version()))

    assert card.ok
    assert card.version == "1"
    assert card.model_family == "HistGradientBoosting"
    assert card.classification_threshold == "0.35"
    assert card.test_metrics["recall"] == 0.6871
    assert card.registered_at == "2026-11-02 02:40 UTC"


def test_seed_model_is_flagged():
    card = build_card(FakeClient(make_version(origen="seed-sintetico")))

    assert card.is_seed
    assert not card.ok
    assert "Modelo semilla sintético" in render_markdown(card)


def test_wrong_family_or_threshold_fails():
    card = build_card(
        FakeClient(make_version(model_family="LogisticRegression", classification_threshold="0.7"))
    )

    failed = {name for name, passed, _ in card.checks if not passed}
    assert "Familia HistGradientBoosting" in failed
    assert any(name.startswith("classification_threshold") for name in failed)


def test_missing_lineage_fails():
    card = build_card(FakeClient(make_version(git_commit=None)))

    assert not card.ok


def test_markdown_contains_lineage_and_metrics():
    text = render_markdown(build_card(FakeClient(make_version())))

    assert RUN_ID in text
    assert "`cb67db696e9316e41e4dc20929f`" in text
    assert "| Recall | F1 | ROC-AUC |" in text
    assert "todas las verificaciones pasan" in text


def test_other_run_with_same_config_fails():
    """Un registro local con otro run (misma configuración) no es el champion documentado."""
    version = make_version(source_run_id="1da3cd94b43e4eeaa1612ea5d02c0456")
    version.run_id = "1da3cd94b43e4eeaa1612ea5d02c0456"
    version.version = "5"
    card = build_card(FakeClient(version))

    failed = {name for name, passed, _ in card.checks if not passed}
    assert failed == {"Es el run documentado en docs/p2_ficha_champion.md"}
