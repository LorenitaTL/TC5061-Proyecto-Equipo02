"""FinOps utilities for MiBici inference workloads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


DEFAULT_PRICES = (
    Path(__file__).resolve().parents[2]
    / "references"
    / "finops_precios.json"
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "reports"
    / "finops"
)

DEFAULT_RPS = 10.0
DEFAULT_UTILIZATION = 0.70
DEFAULT_DAILY_VOLUME = 1000
DEFAULT_DAYS_PER_MONTH = 30


def _validate_positive(name: str, value: float) -> None:
    """Validate that a numeric argument is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_utilization(utilization: float) -> None:
    """Validate utilization is between 0 and 1."""
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be greater than 0 and <= 1")


def cost_per_inference(
    price_per_hour: float,
    rps: float,
    utilization: float = 1.0,
) -> float:
    """
    Calculate the cost of one inference.

    Parameters
    ----------
    price_per_hour:
        Instance price in USD/hour.
    rps:
        Measured requests/inferences per second for one replica.
    utilization:
        Fraction of the measured capacity considered usable.

    Returns
    -------
    float
        USD per inference.
    """
    _validate_positive("price_per_hour", price_per_hour)
    _validate_positive("rps", rps)
    _validate_utilization(utilization)

    effective_rps = rps * utilization

    return price_per_hour / (effective_rps * 3600)


def cost_per_1k(
    price_per_hour: float,
    rps: float,
    utilization: float = 1.0,
) -> float:
    """Calculate the cost of 1,000 inferences."""
    return cost_per_inference(
        price_per_hour=price_per_hour,
        rps=rps,
        utilization=utilization,
    ) * 1000


def monthly_cost(
    daily_volume: float,
    price_per_hour: float,
    rps: float,
    utilization: float = 1.0,
    days: int = DEFAULT_DAYS_PER_MONTH,
) -> float:
    """
    Estimate monthly compute cost for a daily inference volume.

    This model assumes compute is charged according to the actual
    inference volume rather than keeping the instance running 24/7.
    """
    _validate_positive("daily_volume", daily_volume)
    _validate_positive("price_per_hour", price_per_hour)
    _validate_positive("rps", rps)
    _validate_utilization(utilization)

    if days <= 0:
        raise ValueError("days must be greater than zero")

    return (
        daily_volume
        * days
        * cost_per_inference(
            price_per_hour=price_per_hour,
            rps=rps,
            utilization=utilization,
        )
    )


def replicas_needed(
    target_rps: float,
    rps_per_replica: float,
) -> int:
    """
    Calculate the number of replicas required for a target RPS.
    """
    _validate_positive("target_rps", target_rps)
    _validate_positive("rps_per_replica", rps_per_replica)

    return math.ceil(target_rps / rps_per_replica)


def resource_cost_per_hour(
    vcpu: float,
    memory_gb: float,
    vcpu_price_per_hour: float,
    gb_price_per_hour: float,
) -> float:
    """
    Calculate resource-based hourly cost.

    The function treats vCPU and GB-hour prices as independent
    unit prices.
    """
    _validate_positive("vcpu", vcpu)
    _validate_positive("memory_gb", memory_gb)
    _validate_positive("vcpu_price_per_hour", vcpu_price_per_hour)
    _validate_positive("gb_price_per_hour", gb_price_per_hour)

    return (
        vcpu * vcpu_price_per_hour
        + memory_gb * gb_price_per_hour
    )


def load_prices(path: str | Path = DEFAULT_PRICES) -> dict:
    """Load instance prices from the JSON reference file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "instances" not in data:
        raise ValueError("Price file must contain an 'instances' list")

    return data


def build_cost_report(
    prices: dict,
    rps: float = DEFAULT_RPS,
    utilization: float = DEFAULT_UTILIZATION,
    daily_volume: int = DEFAULT_DAILY_VOLUME,
) -> pd.DataFrame:
    """
    Build a comparison table for all reference instances.
    """
    rows = []

    for instance in prices["instances"]:
        price = float(instance["price_per_hour"])

        rows.append(
            {
                "provider": instance["provider"],
                "instancia": instance["name"],
                "region": instance["region"],
                "vcpu": instance["vcpu"],
                "memory_gb": instance["memory_gb"],
                "precio_usd_h": price,
                "rps": rps,
                "utilizacion": utilization,
                "costo_inferencia_usd": cost_per_inference(
                    price,
                    rps,
                    utilization,
                ),
                "costo_1k_usd": cost_per_1k(
                    price,
                    rps,
                    utilization,
                ),
                "costo_mensual_usd": monthly_cost(
                    daily_volume,
                    price,
                    rps,
                    utilization,
                ),
                "replicas_10_rps": replicas_needed(
                    10,
                    rps * utilization,
                ),
            }
        )

    return pd.DataFrame(rows)


def write_markdown_report(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    prices: dict,
    rps: float,
    utilization: float,
    daily_volume: int,
) -> None:
    """Write the FinOps report as Markdown."""
    output_path = Path(output_path)

    lines = [
        "# Análisis FinOps",
        "",
        "## Supuestos",
        "",
        f"- RPS por réplica: **{rps:g}**",
        f"- Utilización considerada: **{utilization:.0%}**",
        f"- Volumen diario de referencia: **{daily_volume:,} inferencias**",
        f"- Días por mes: **{DEFAULT_DAYS_PER_MONTH}**",
        "- Moneda: **USD**",
        "- Modelo de precio: **On-Demand**",
        "",
        "## Comparación de instancias",
        "",
    ]

    table = dataframe.copy()

    numeric_columns = [
        "precio_usd_h",
        "costo_inferencia_usd",
        "costo_1k_usd",
        "costo_mensual_usd",
    ]

    for column in numeric_columns:
        table[column] = table[column].map(
            lambda value: f"{value:.8f}"
        )


    headers = list(table.columns)

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in table.iterrows():
        values = [str(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")




    lines.extend(
        [
            "",
            "## Fuentes",
            "",
        ]
    )

    for instance in prices["instances"]:
        lines.append(
            f"- **{instance['provider']} {instance['name']}**: "
            f"{instance['source']} — {instance['source_url']}"
        )

    lines.extend(
        [
            "",
            "## Interpretación",
            "",
            "El costo por inferencia se obtiene dividiendo el costo "
            "horario entre la capacidad efectiva de inferencia. "
            "La capacidad efectiva considera el RPS medido y el "
            "porcentaje de utilización seleccionado.",
            "",
            "Estos valores son estimaciones de cómputo y no incluyen "
            "almacenamiento, transferencia de datos, balanceadores, "
            "bases de datos, observabilidad ni impuestos.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def generate_report(
    prices_path: str | Path = DEFAULT_PRICES,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    rps: float = DEFAULT_RPS,
    utilization: float = DEFAULT_UTILIZATION,
    daily_volume: int = DEFAULT_DAILY_VOLUME,
) -> pd.DataFrame:
    """Generate costos.csv and costos.md."""
    prices = load_prices(prices_path)

    dataframe = build_cost_report(
        prices=prices,
        rps=rps,
        utilization=utilization,
        daily_volume=daily_volume,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "costos.csv"
    markdown_path = output_dir / "costos.md"

    dataframe.to_csv(csv_path, index=False)

    write_markdown_report(
        dataframe=dataframe,
        output_path=markdown_path,
        prices=prices,
        rps=rps,
        utilization=utilization,
        daily_volume=daily_volume,
    )

    return dataframe


def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate MiBici FinOps cost estimates."
    )

    parser.add_argument(
        "--prices",
        type=Path,
        default=DEFAULT_PRICES,
        help="Path to finops_precios.json",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )

    parser.add_argument(
        "--rps",
        type=float,
        default=DEFAULT_RPS,
        help="RPS per replica",
    )

    parser.add_argument(
        "--utilization",
        type=float,
        default=DEFAULT_UTILIZATION,
        help="Expected utilization from 0 to 1",
    )

    parser.add_argument(
        "--daily-volume",
        type=int,
        default=DEFAULT_DAILY_VOLUME,
        help="Daily inference volume",
    )

    args = parser.parse_args()

    dataframe = generate_report(
        prices_path=args.prices,
        output_dir=args.output_dir,
        rps=args.rps,
        utilization=args.utilization,
        daily_volume=args.daily_volume,
    )

    print(dataframe.to_string(index=False))
    print(f"\nReportes generados en: {args.output_dir}")


if __name__ == "__main__":
    main()