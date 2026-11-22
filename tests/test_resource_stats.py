"""Tests for Docker resource statistics utilities."""

import json

import pandas as pd
import pytest

from mi_bici.operations.resource_stats import (
    generate_report,
    parse_stats,
    summarize,
)


def _write_stats_file(path):
    records = [
        {
            "ts": "2026-11-21T10:00:00Z",
            "stats": {
                "Name": "mibici-api-1",
                "CPUPerc": "10.00%",
                "MemUsage": "200MiB / 1GiB",
                "MemPerc": "19.53%",
                "NetIO": "1MB / 2kB",
            },
        },
        {
            "ts": "2026-11-21T10:00:02Z",
            "stats": {
                "Name": "mibici-api-1",
                "CPUPerc": "20.00%",
                "MemUsage": "300MiB / 1GiB",
                "MemPerc": "29.30%",
                "NetIO": "2MB / 4kB",
            },
        },
        {
            "ts": "2026-11-21T10:00:04Z",
            "stats": {
                "Name": "mibici-api-1",
                "CPUPerc": "30.00%",
                "MemUsage": "400MiB / 1GiB",
                "MemPerc": "39.06%",
                "NetIO": "3MB / 6kB",
            },
        },
        {
            "ts": "2026-11-21T10:00:00Z",
            "stats": {
                "Name": "mlflow",
                "CPUPerc": "5.00%",
                "MemUsage": "100MiB / 1GiB",
                "MemPerc": "9.77%",
                "NetIO": "1MB / 2kB",
            },
        },
        {
            "ts": "2026-11-21T10:00:02Z",
            "stats": {
                "Name": "mlflow",
                "CPUPerc": "15.00%",
                "MemUsage": "200MiB / 1GiB",
                "MemPerc": "19.53%",
                "NetIO": "2MB / 4kB",
            },
        },
    ]

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def test_parse_stats(tmp_path):
    stats_file = tmp_path / "stats.jsonl"

    _write_stats_file(stats_file)

    df = parse_stats(stats_file)

    assert len(df) == 5

    assert set(df["contenedor"]) == {
        "mibici-api-1",
        "mlflow",
    }

    assert df.loc[0, "cpu_pct"] == pytest.approx(10.0)

    assert df.loc[0, "mem_mib"] == pytest.approx(200.0)

    assert df.loc[0, "mem_limit_mib"] == pytest.approx(1024.0)

    assert df.loc[0, "mem_pct"] == pytest.approx(19.53)


def test_summarize():
    df = pd.DataFrame(
        {
            "contenedor": [
                "api",
                "api",
                "api",
            ],
            "cpu_pct": [
                10.0,
                20.0,
                30.0,
            ],
            "mem_mib": [
                200.0,
                300.0,
                400.0,
            ],
            "mem_limit_mib": [
                1024.0,
                1024.0,
                1024.0,
            ],
        }
    )

    summary = summarize(df)

    assert len(summary) == 1

    row = summary.iloc[0]

    assert row["contenedor"] == "api"

    assert row["cpu_pct_prom"] == pytest.approx(20.0)

    assert row["cpu_pct_p95"] == pytest.approx(29.0)

    assert row["mem_mib_prom"] == pytest.approx(300.0)

    assert row["mem_mib_p95"] == pytest.approx(390.0)

    assert row["mem_limite_mib"] == pytest.approx(1024.0)


def test_generate_report(tmp_path):
    stats_file = tmp_path / "stats.jsonl"
    output_dir = tmp_path / "output"

    _write_stats_file(stats_file)

    summary = generate_report(
        jsonl=stats_file,
        out_dir=output_dir,
    )

    assert len(summary) == 2

    assert (output_dir / "stats_resumen.csv").exists()

    assert (output_dir / "figures" / "cpu.png").exists()

    assert (output_dir / "figures" / "memory.png").exists()
