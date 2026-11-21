import random

from pydantic import ValidationError
import pytest

from mi_bici.api.schemas import PredictionRequest
from mi_bici.operations.load_payloads import make_invalid_payload, make_payload


@pytest.mark.parametrize("seed", range(100))
def test_payload_es_valido_y_coherente(seed):
    payload = make_payload(random.Random(seed))

    PredictionRequest.model_validate(payload)
    assert len(payload) == 30
    assert payload["net_flow"] == payload["departures"] - payload["arrivals"]
    assert payload["absolute_imbalance"] == abs(payload["net_flow"])
    assert payload["imbalance_ratio"] == pytest.approx(
        payload["absolute_imbalance"] / payload["capacity"]
    )
    assert payload["is_weekend"] == int(payload["day_of_week"] >= 5)


def test_payload_es_reproducible_con_la_misma_semilla():
    assert make_payload(random.Random(42)) == make_payload(random.Random(42))


def test_payload_invalido_falla_por_la_hora():
    payload = make_invalid_payload(random.Random(42))

    with pytest.raises(ValidationError) as error:
        PredictionRequest.model_validate(payload)

    assert any(issue["loc"] == ("hour_of_day",) for issue in error.value.errors())
