"""Tests for FinOps calculations."""

import json

import pytest

from mi_bici.operations.finops import (
    build_cost_report,
    cost_per_1k,
    cost_per_inference,
    monthly_cost,
    replicas_needed,
    resource_cost_per_hour,
)


def test_cost_per_inference():
    # $36/hour, 10 RPS, 100% utilization.
    #
    # 10 * 3600 = 36,000 inferences/hour
    # $36 / 36,000 = $0.001/inference
    result = cost_per_inference(
        price_per_hour=36,
        rps=10,
        utilization=1.0,
    )

    assert result == pytest.approx(0.001)


def test_cost_per_1k():
    result = cost_per_1k(
        price_per_hour=36,
        rps=10,
        utilization=1.0,
    )

    assert result == pytest.approx(1.0)


def test_cost_per_inference_with_utilization():
    # Effective capacity:
    # 10 RPS * 50% = 5 RPS
    #
    # 5 * 3600 = 18,000 inferences/hour
    # $18 / 18,000 = $0.001
    result = cost_per_inference(
        price_per_hour=18,
        rps=10,
        utilization=0.5,
    )

    assert result == pytest.approx(0.001)


def test_monthly_cost():
    # 1,000 inferences/day
    # 30 days
    # $0.001/inference
    result = monthly_cost(
        daily_volume=1000,
        price_per_hour=36,
        rps=10,
        utilization=1.0,
        days=30,
    )

    assert result == pytest.approx(30.0)


def test_replicas_needed():
    assert replicas_needed(
        target_rps=25,
        rps_per_replica=10,
    ) == 3

    assert replicas_needed(
        target_rps=20,
        rps_per_replica=10,
    ) == 2


def test_resource_cost_per_hour():
    result = resource_cost_per_hour(
        vcpu=2,
        memory_gb=4,
        vcpu_price_per_hour=0.01,
        gb_price_per_hour=0.005,
    )

    assert result == pytest.approx(0.04)


def test_build_cost_report():
    prices = {
        "instances": [
            {
                "provider": "TEST",
                "name": "test.small",
                "region": "test",
                "price_per_hour": 36.0,
                "vcpu": 2,
                "memory_gb": 4,
                "source": "synthetic",
                "source_url": "https://example.com",
            }
        ]
    }

    report = build_cost_report(
        prices=prices,
        rps=10,
        utilization=1.0,
        daily_volume=1000,
    )

    assert len(report) == 1

    row = report.iloc[0]

    assert row["instancia"] == "test.small"
    assert row["costo_inferencia_usd"] == pytest.approx(0.001)
    assert row["costo_1k_usd"] == pytest.approx(1.0)
    assert row["costo_mensual_usd"] == pytest.approx(30.0)
    assert row["replicas_10_rps"] == 1