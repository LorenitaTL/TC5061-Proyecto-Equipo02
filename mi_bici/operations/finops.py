"""FinOps utilities for MiBici inference workloads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

DEFAULT_PRICES = Path(__file__).resolve().parents[2] / "references" / "finops_precios.json"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "reports" / "finops"

DEFAULT_LOAD_TEST_DIR = Path(__file__).resolve().parents[2] / "reports" / "load_test"


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
    return (
        cost_per_inference(
            price_per_hour=price_per_hour,
            rps=rps,
            utilization=utilization,
        )
        * 1000
    )


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

    return vcpu * vcpu_price_per_hour + memory_gb * gb_price_per_hour


def load_prices(path: str | Path = DEFAULT_PRICES) -> dict:
    """Load instance prices from the JSON reference file."""
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "instances" not in data:
        raise ValueError("Price file must contain an 'instances' list")

    return data


def load_load_test_results(
    load_test_dir: str | Path,
) -> pd.DataFrame:
    """
    Load M3 load-test results and E1 resource summaries.

    Expected structure:
        load_test_dir/
            resumen.csv
            u10/stats_resumen.csv
            u50/stats_resumen.csv
            u100/stats_resumen.csv
    """
    load_test_dir = Path(load_test_dir)

    resumen_path = load_test_dir / "resumen.csv"

    if not resumen_path.exists():
        raise FileNotFoundError(f"Load-test summary not found: {resumen_path}")

    performance = pd.read_csv(resumen_path)

    required_columns = {
        "escenario",
        "usuarios",
        "rps",
        "p95_ms",
        "p99_ms",
    }

    missing = required_columns - set(performance.columns)

    if missing:
        raise ValueError(f"resumen.csv is missing columns: {sorted(missing)}")

    rows = []

    for _, result in performance.iterrows():
        escenario = result["escenario"]

        stats_path = load_test_dir / str(escenario) / "stats_resumen.csv"

        if not stats_path.exists():
            raise FileNotFoundError(f"Resource summary not found: {stats_path}")

        stats = pd.read_csv(stats_path)

        api = stats[stats["contenedor"] == "mibici-api-1"]

        if api.empty:
            raise ValueError(f"mibici-api-1 not found in {stats_path}")

        api = api.iloc[0]

        rows.append(
            {
                "escenario": escenario,
                "usuarios": int(result["usuarios"]),
                "rps": float(result["rps"]),
                "p50_ms": float(result["p50_ms"]),
                "p95_ms": float(result["p95_ms"]),
                "p99_ms": float(result["p99_ms"]),
                "tasa_error": float(result["tasa_error"]),
                "cpu_pct_prom": float(api["cpu_pct_prom"]),
                "cpu_pct_p95": float(api["cpu_pct_p95"]),
                "mem_mib_prom": float(api["mem_mib_prom"]),
                "mem_mib_p95": float(api["mem_mib_p95"]),
                "mem_limite_mib": float(api["mem_limite_mib"]),
            }
        )

    return pd.DataFrame(rows)


def monthly_instance_cost(
    price_per_hour: float,
    replicas: int = 1,
    hours_per_month: float = 730,
) -> float:
    """Calcula el costo mensual de mantener instancias encendidas 24/7."""
    _validate_positive("price_per_hour", price_per_hour)
    _validate_positive("replicas", replicas)
    _validate_positive("hours_per_month", hours_per_month)

    return price_per_hour * replicas * hours_per_month


def build_e3_finops_report(
    load_test_dir: Path = DEFAULT_LOAD_TEST_DIR,
    price_per_hour: float = 0.0416,
    daily_volume: int = 8976,
    stress_multiplier: float = 10.0,
    utilization: float = 1.0,
) -> pd.DataFrame:
    """
    Construye el análisis FinOps de E3 utilizando los resultados reales
    de los escenarios de carga M3 y las métricas de recursos de E1.

    El volumen diario se estima como:
        estaciones * predicciones por estación por día

    Para este proyecto:
        374 estaciones * 24 predicciones/día = 8,976 inferencias/día

    El escenario de estrés utiliza 10x ese volumen.

    Las réplicas se calculan a partir del RPS promedio requerido para
    procesar el volumen diario y del RPS medido por escenario.
    """

    load_test_dir = Path(load_test_dir)

    # Resultados reales de M3 + E1
    load_results = load_load_test_results(load_test_dir)

    if load_results.empty:
        raise ValueError(f"No se encontraron resultados de carga en {load_test_dir}")

    # Validaciones

    # Validaciones
    _validate_positive("price_per_hour", price_per_hour)
    _validate_positive("daily_volume", daily_volume)
    _validate_positive("stress_multiplier", stress_multiplier)

    _validate_utilization(utilization)

    # Parámetros de volumen
    days_per_month = DEFAULT_DAYS_PER_MONTH

    monthly_volume = daily_volume * days_per_month

    daily_volume_10x = daily_volume * stress_multiplier
    monthly_volume_10x = monthly_volume * stress_multiplier

    # Convertimos volumen diario a RPS promedio requerido.
    seconds_per_day = 24 * 60 * 60

    target_rps = daily_volume / seconds_per_day
    target_rps_10x = daily_volume_10x / seconds_per_day

    rows = []

    for _, result in load_results.iterrows():
        measured_rps = float(result["rps"])

        # Costo unitario basado en la capacidad medida.
        cost_inference = cost_per_inference(
            price_per_hour=price_per_hour,
            rps=measured_rps,
            utilization=utilization,
        )

        # Réplicas necesarias para soportar el volumen promedio 10x.
        replicas_10x = replicas_needed(
            target_rps=target_rps_10x,
            rps_per_replica=measured_rps,
        )

        # Costo mensual si las réplicas permanecen encendidas 24/7.
        cost_24x7_month = monthly_instance_cost(
            price_per_hour=price_per_hour,
            replicas=replicas_10x,
            hours_per_month=730,
        )

        rows.append(
            {
                "escenario": result["escenario"],
                "usuarios": result["usuarios"],
                "rps": measured_rps,
                "p50_ms": result["p50_ms"],
                "p95_ms": result["p95_ms"],
                "p99_ms": result["p99_ms"],
                "tasa_error": result["tasa_error"],
                "cpu_pct_prom": result["cpu_pct_prom"],
                "cpu_pct_p95": result["cpu_pct_p95"],
                "mem_mib_prom": result["mem_mib_prom"],
                "mem_mib_p95": result["mem_mib_p95"],
                "mem_limite_mib": result["mem_limite_mib"],
                # Volumen MiBici
                "inferencias_dia": daily_volume,
                "inferencias_mes": monthly_volume,
                "inferencias_dia_10x": daily_volume_10x,
                "inferencias_mes_10x": monthly_volume_10x,
                # RPS requerido
                "target_rps": target_rps,
                "target_rps_10x": target_rps_10x,
                # Costos unitarios
                "costo_inferencia_usd": cost_inference,
                "costo_1000_usd": cost_per_1k(
                    price_per_hour=price_per_hour,
                    rps=measured_rps,
                    utilization=utilization,
                ),
                # Costo basado en volumen de inferencias
                "costo_mes_usd": (monthly_volume * cost_inference),
                "costo_mes_10x_usd": (monthly_volume_10x * cost_inference),
                # Escalabilidad real
                "replicas_10x": replicas_10x,
                # Infraestructura encendida 24/7
                "costo_24x7_mes_usd": cost_24x7_month,
            }
        )

    return pd.DataFrame(rows)


def write_e3_markdown_report(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write the E3 FinOps analysis as Markdown."""
    output_path = Path(output_path)

    lines = [
        "# Análisis FinOps E3",
        "",
        "## Supuestos",
        "",
        "- Estaciones de referencia: **374**",
        "- Predicciones por estación: **24/día**",
        "- Volumen de referencia: **8,976 inferencias/día**",
        "- Escenario de estrés: **10x**",
        "- Volumen de estrés: **89,760 inferencias/día**",
        "- Días por mes: **30**",
        "- Horas de operación mensual para costo 24/7: **730 h**",
        "- Moneda: **USD**",
        "- Capacidad utilizada: RPS medido en cada escenario de M3",
        "",
        "## Resultados de rendimiento y recursos",
        "",
    ]

    # Columnas principales para el análisis E3.
    # Se excluyen columnas auxiliares para mantener el reporte legible.
    report_columns = [
        "escenario",
        "usuarios",
        "rps",
        "p95_ms",
        "cpu_pct_prom",
        "cpu_pct_p95",
        "mem_mib_prom",
        "mem_mib_p95",
        "tasa_error",
    ]

    performance_table = dataframe[report_columns].copy()

    numeric_columns = [
        "rps",
        "p95_ms",
        "cpu_pct_prom",
        "cpu_pct_p95",
        "mem_mib_prom",
        "mem_mib_p95",
        "tasa_error",
    ]

    for column in numeric_columns:
        performance_table[column] = performance_table[column].map(lambda value: f"{value:.4f}")

    headers = list(performance_table.columns)

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in performance_table.iterrows():
        values = [str(row[column]) for column in headers]

        lines.append("| " + " | ".join(values) + " |")

    # ---------------------------------------------------------
    # FinOps
    # ---------------------------------------------------------

    lines.extend(
        [
            "",
            "## Análisis FinOps",
            "",
            (
                "El volumen de referencia corresponde a 374 estaciones "
                "con una predicción horaria por estación:"
            ),
            "",
            "**374 × 24 = 8,976 inferencias/día.**",
            "",
            "Para el escenario de estrés de 10x:",
            "",
            "**8,976 × 10 = 89,760 inferencias/día.**",
            "",
            (
                "El RPS promedio requerido para procesar este volumen "
                "continuamente durante 24 horas es aproximadamente:"
            ),
            "",
            "**89,760 / 86,400 = 1.0389 RPS.**",
            "",
            (
                "Las réplicas necesarias se calculan comparando este "
                "RPS objetivo con el RPS medido experimentalmente "
                "para cada escenario de carga."
            ),
            "",
        ]
    )

    # Tabla FinOps
    finops_columns = [
        "escenario",
        "rps",
        "target_rps_10x",
        "replicas_10x",
        "costo_inferencia_usd",
        "costo_1000_usd",
        "costo_mes_10x_usd",
        "costo_24x7_mes_usd",
    ]

    finops_table = dataframe[finops_columns].copy()

    numeric_columns = [
        "rps",
        "target_rps_10x",
        "costo_inferencia_usd",
        "costo_1000_usd",
        "costo_mes_10x_usd",
        "costo_24x7_mes_usd",
    ]

    for column in numeric_columns:
        if column == "costo_inferencia_usd":
            finops_table[column] = finops_table[column].map(lambda value: f"{value:.8f}")
        else:
            finops_table[column] = finops_table[column].map(lambda value: f"{value:.6f}")

    headers = list(finops_table.columns)

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in finops_table.iterrows():
        values = [str(row[column]) for column in headers]

        lines.append("| " + " | ".join(values) + " |")

    # ---------------------------------------------------------
    # Interpretación
    # ---------------------------------------------------------

    lines.extend(
        [
            "",
            "## Interpretación",
            "",
            "El escenario u10 alcanza aproximadamente 6.21 RPS con un P95 de latencia de 50 ms.",
            "",
            "El escenario u50 alcanza aproximadamente 27.88 RPS con un P95 de latencia de 620 ms.",
            "",
            (
                "Al aumentar la carga hasta u100, el throughput "
                "solo aumenta hasta aproximadamente 29.68 RPS, "
                "mientras que el P95 aumenta hasta 2,500 ms."
            ),
            "",
            (
                "Esto indica que la API comienza a saturarse "
                "alrededor de 30 RPS bajo la configuración evaluada."
            ),
            "",
            (
                "El consumo de CPU de la API aumenta de forma "
                "importante con la carga: el promedio pasa de "
                "15.7% en u10 a 66.3% en u50 y 93.1% en u100."
            ),
            "",
            (
                "El CPU p95 alcanza aproximadamente 107.2% en u100, "
                "lo que indica que la configuración actual se "
                "encuentra bajo presión de CPU al acercarse a "
                "30 RPS."
            ),
            "",
            (
                "El consumo de memoria de la API permanece "
                "relativamente estable, alrededor de 224–242 MiB "
                "promedio, por lo que CPU y latencia constituyen "
                "los principales indicadores de saturación de la API."
            ),
            "",
            (
                "El contenedor de MLflow presenta un consumo de "
                "memoria cercano al límite configurado de 1 GiB. "
                "Esto representa un riesgo de capacidad que debe "
                "considerarse en futuras optimizaciones, aunque "
                "los datos disponibles no permiten atribuirle "
                "directamente la latencia observada en la API."
            ),
            "",
            (
                "El volumen de 8,976 inferencias/día es un supuesto "
                "de dimensionamiento basado en 374 estaciones y "
                "una predicción horaria por estación; no representa "
                "un volumen oficial declarado por MiBici."
            ),
            "",
            (
                "El cálculo de costo 24/7 representa el costo de "
                "mantener las réplicas encendidas durante 730 horas "
                "mensuales. El costo por inferencia representa una "
                "métrica de eficiencia basada en el RPS medido y "
                "no debe interpretarse como el costo de una instancia "
                "encendida continuamente."
            ),
            "",
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


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
        table[column] = table[column].map(lambda value: f"{value:.8f}")

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
            (
                "El costo por inferencia se obtiene dividiendo el costo "
                "horario entre la capacidad efectiva de inferencia. "
                "La capacidad efectiva considera el RPS medido y el "
                "porcentaje de utilización seleccionado."
            ),
            "",
            (
                "Estos valores son estimaciones de cómputo y no incluyen "
                "almacenamiento, transferencia de datos, balanceadores, "
                "bases de datos, observabilidad ni impuestos."
            ),
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
    parser = argparse.ArgumentParser(description="Generate MiBici FinOps cost estimates.")

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

    parser.add_argument(
        "--e3",
        action="store_true",
        help="Generate the E3 FinOps report using M3/E1 results",
    )

    parser.add_argument(
        "--load-test-dir",
        type=Path,
        default=(Path(__file__).resolve().parents[2] / "reports" / "load_test"),
        help="Directory containing M3/E1 load-test results",
    )

    parser.add_argument(
        "--e3-price-per-hour",
        type=float,
        default=0.0416,
        help="Hourly instance price used for E3",
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

    if args.e3:
        e3_dataframe = build_e3_finops_report(
            load_test_dir=args.load_test_dir,
            price_per_hour=args.e3_price_per_hour,
        )

        e3_csv_path = args.output_dir / "e3_finops.csv"

        e3_md_path = args.output_dir / "e3_finops.md"

        e3_dataframe.to_csv(
            e3_csv_path,
            index=False,
        )

        write_e3_markdown_report(
            dataframe=e3_dataframe,
            output_path=e3_md_path,
        )

        print("\nReporte E3:")
        print(e3_dataframe.to_string(index=False))
        print(f"\nE3 CSV generado en: {e3_csv_path}")
        print(f"E3 Markdown generado en: {e3_md_path}")

    print(f"\nReportes generados en: {args.output_dir}")


if __name__ == "__main__":
    main()
