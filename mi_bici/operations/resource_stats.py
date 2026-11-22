"""Utilities to parse and summarize Docker resource statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

_UNIT_TO_MIB = {
    "b": 1 / (1024**2),
    "kb": 1 / 1024,
    "kib": 1 / 1024,
    "mb": 1 / 1024,
    "mib": 1,
    "gb": 1024 / 1,
    "gib": 1024,
}


def _parse_size_to_mib(value: str | float | None) -> float:
    """Convert a Docker size string to MiB."""
    if value is None:
        return float("nan")

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if not text:
        return float("nan")

    match = re.match(
        r"^\s*([-+]?\d*\.?\d+)\s*([a-zA-Z]+)\s*$",
        text,
    )

    if not match:
        raise ValueError(f"Could not parse size value: {value!r}")

    number = float(match.group(1))
    unit = match.group(2).lower()

    if unit not in _UNIT_TO_MIB:
        raise ValueError(f"Unsupported size unit: {unit!r}")

    return number * _UNIT_TO_MIB[unit]


def _parse_percentage(value: str | float | None) -> float:
    """Convert a percentage such as '15.40%' to a float."""
    if value is None:
        return float("nan")

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace("%", "")

    if not text:
        return float("nan")

    return float(text)


def _parse_memory_usage(value: str) -> tuple[float, float]:
    """
    Parse Docker's MemUsage field.

    Example:
        '216.5MiB / 1GiB'

    Returns:
        (memory_used_mib, memory_limit_mib)
    """
    if not value or "/" not in value:
        return float("nan"), float("nan")

    used, limit = value.split("/", maxsplit=1)

    return (
        _parse_size_to_mib(used.strip()),
        _parse_size_to_mib(limit.strip()),
    )


def _parse_network(value: str) -> tuple[float, float]:
    """
    Parse Docker's NetIO field.

    Example:
        '1.79MB / 12.9kB'

    Returns:
        (received_mib, transmitted_mib)
    """
    if not value or "/" not in value:
        return float("nan"), float("nan")

    received, transmitted = value.split("/", maxsplit=1)

    return (
        _parse_size_to_mib(received.strip()),
        _parse_size_to_mib(transmitted.strip()),
    )


def parse_stats(jsonl: str | Path) -> pd.DataFrame:
    """
    Parse a Docker stats JSONL file into a DataFrame.

    Expected format per line:

        {
            "ts": "...",
            "stats": {
                "Name": "mibici-api-1",
                "CPUPerc": "15.40%",
                "MemUsage": "216.5MiB / 1GiB",
                "MemPerc": "21.14%",
                "NetIO": "1.79MB / 12.9kB"
            }
        }

    Returns a DataFrame with one row per container/sample.
    """
    path = Path(jsonl)

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc

            stats = record.get("stats", {})

            if not stats:
                raise ValueError(f"Missing 'stats' object in {path} at line {line_number}")

            memory_mib, memory_limit_mib = _parse_memory_usage(stats.get("MemUsage"))

            net_rx_mib, net_tx_mib = _parse_network(stats.get("NetIO"))

            rows.append(
                {
                    "timestamp": pd.to_datetime(
                        record.get("ts"),
                        utc=True,
                    ),
                    "contenedor": stats.get("Name"),
                    "cpu_pct": _parse_percentage(stats.get("CPUPerc")),
                    "mem_mib": memory_mib,
                    "mem_pct": _parse_percentage(stats.get("MemPerc")),
                    "mem_limit_mib": memory_limit_mib,
                    "net_rx_mib": net_rx_mib,
                    "net_tx_mib": net_tx_mib,
                }
            )

    columns = [
        "timestamp",
        "contenedor",
        "cpu_pct",
        "mem_mib",
        "mem_pct",
        "mem_limit_mib",
        "net_rx_mib",
        "net_tx_mib",
    ]

    return pd.DataFrame(rows, columns=columns)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average and p95 CPU/memory per container.

    Output columns follow the Activity 9 contract:

        contenedor
        cpu_pct_prom
        cpu_pct_p95
        mem_mib_prom
        mem_mib_p95
        mem_limite_mib
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "contenedor",
                "cpu_pct_prom",
                "cpu_pct_p95",
                "mem_mib_prom",
                "mem_mib_p95",
                "mem_limite_mib",
            ]
        )

    summary = (
        df.groupby("contenedor", dropna=False)
        .agg(
            cpu_pct_prom=("cpu_pct", "mean"),
            cpu_pct_p95=("cpu_pct", lambda x: x.quantile(0.95)),
            mem_mib_prom=("mem_mib", "mean"),
            mem_mib_p95=("mem_mib", lambda x: x.quantile(0.95)),
            mem_limite_mib=("mem_limit_mib", "max"),
        )
        .reset_index()
    )

    return summary


def plot_stats(
    df: pd.DataFrame,
    out_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Generate CPU and memory usage plots by container.

    Returns:
        (cpu_plot_path, memory_plot_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_path = out_dir / "cpu.png"
    memory_path = out_dir / "memory.png"

    # CPU
    fig, ax = plt.subplots(figsize=(10, 5))

    for container, group in df.groupby("contenedor"):
        group = group.sort_values("timestamp")
        ax.plot(
            group["timestamp"],
            group["cpu_pct"],
            label=container,
        )

    ax.set_title("CPU por contenedor")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("CPU (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(cpu_path, dpi=150)
    plt.close(fig)

    # Memory
    fig, ax = plt.subplots(figsize=(10, 5))

    for container, group in df.groupby("contenedor"):
        group = group.sort_values("timestamp")
        ax.plot(
            group["timestamp"],
            group["mem_mib"],
            label=container,
        )

    ax.set_title("Memoria por contenedor")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Memoria (MiB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(memory_path, dpi=150)
    plt.close(fig)

    return cpu_path, memory_path


def generate_report(
    jsonl: str | Path,
    out_dir: str | Path,
) -> pd.DataFrame:
    """Parse stats, create plots and save the summary CSV."""
    out_dir = Path(out_dir)

    df = parse_stats(jsonl)

    summary = summarize(df)

    summary_path = out_dir / "stats_resumen.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_path, index=False)

    plot_stats(df, out_dir / "figures")

    return summary


def main() -> None:
    """Command-line interface for resource statistics."""
    parser = argparse.ArgumentParser(description="Parse and summarize Docker resource statistics.")

    parser.add_argument(
        "jsonl",
        type=Path,
        help="Path to stats.jsonl",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for summary and figures",
    )

    args = parser.parse_args()

    summary = generate_report(
        jsonl=args.jsonl,
        out_dir=args.out_dir,
    )

    print(summary.to_string(index=False))
    print(f"\nReporte generado en: {args.out_dir}")


if __name__ == "__main__":
    main()
