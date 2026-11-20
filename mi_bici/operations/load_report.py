"""Lectura y resumen de resultados de carga de Locust."""

import csv
from pathlib import Path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def summarize_run(
    stats_path: str | Path,
    history_path: str | Path,
    scenario: str,
) -> dict[str, str | int | float]:
    """Resume las predicciones válidas de una ejecución de Locust."""
    stats = read_csv(stats_path)
    history = read_csv(history_path)

    predictions = [row for row in stats if row["Type"] == "POST" and row["Name"] == "/predict"]
    if len(predictions) != 1:
        raise ValueError("Se esperaba una fila POST /predict en stats.csv")

    row = predictions[0]
    timestamps = [int(item["Timestamp"]) for item in history]
    users = [int(item["User Count"]) for item in history]

    if not timestamps:
        raise ValueError("El historial de Locust está vacío")

    requests = int(row["Request Count"])
    failures = int(row["Failure Count"])

    return {
        "escenario": scenario,
        "usuarios": max(users),
        "duracion_s": max(timestamps) - min(timestamps),
        "peticiones": requests,
        "fallas": failures,
        "tasa_error": failures / requests if requests else 0.0,
        "rps": float(row["Requests/s"]),
        "p50_ms": float(row["50%"]),
        "p95_ms": float(row["95%"]),
        "p99_ms": float(row["99%"]),
    }


COLUMNS = (
    "escenario",
    "usuarios",
    "duracion_s",
    "peticiones",
    "fallas",
    "tasa_error",
    "rps",
    "p50_ms",
    "p95_ms",
    "p99_ms",
)


def write_summary(
    runs_dir: str | Path,
    output_dir: str | Path,
) -> list[dict[str, str | int | float]]:
    """Genera resumen.csv y resumen.md para los escenarios disponibles."""
    runs_dir = Path(runs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        summarize_run(
            scenario_dir / "locust_stats.csv",
            scenario_dir / "locust_stats_history.csv",
            scenario_dir.name,
        )
        for scenario_dir in sorted(runs_dir.iterdir())
        if scenario_dir.is_dir()
        and (scenario_dir / "locust_stats.csv").is_file()
        and (scenario_dir / "locust_stats_history.csv").is_file()
    ]

    if not rows:
        raise ValueError(f"No hay escenarios en {runs_dir}")

    with (output_dir / "resumen.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Resumen de pruebas de carga",
        "",
        (
            "Las métricas de solicitudes corresponden únicamente a POST /predict "
            "con payload válido. La duración es el intervalo observado en el "
            "historial de Locust."
        ),
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for row in rows:
        values = [
            f"{row[column]:.4f}" if column in {"tasa_error", "rps"} else str(row[column])
            for column in COLUMNS
        ]
        lines.append("| " + " | ".join(values) + " |")

    (output_dir / "resumen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def make_plots(runs_dir: str | Path, output_dir: str | Path) -> None:
    """Grafica latencias y RPS por escenario, y tasa de error comparada."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_dir = Path(runs_dir)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries = write_summary(runs_dir, output_dir)
    rps_figure, rps_axis = plt.subplots(figsize=(9, 5))

    for summary in summaries:
        scenario = str(summary["escenario"])
        history = read_csv(runs_dir / scenario / "locust_stats_history.csv")
        points = sorted(
            (row for row in history if row["Type"] == "POST" and row["Name"] == "/predict"),
            key=lambda row: int(row["Timestamp"]),
        )
        if not points:
            raise ValueError(f"Falta el historial POST /predict: {scenario}")

        start = int(points[0]["Timestamp"])
        seconds = [int(row["Timestamp"]) - start for row in points]

        latency_figure, latency_axis = plt.subplots(figsize=(9, 5))
        for percentile in ("50%", "95%", "99%"):
            latency_axis.plot(
                seconds,
                [float(row[percentile] or 0) for row in points],
                label=percentile,
            )

        latency_axis.set(
            title=f"Latencia de /predict: {scenario}",
            xlabel="Tiempo transcurrido (s)",
            ylabel="Latencia (ms)",
        )
        latency_axis.legend(title="Percentil")
        latency_axis.grid(alpha=0.3)
        latency_figure.tight_layout()
        latency_figure.savefig(figures_dir / f"latencia_{scenario}.png", dpi=150)
        plt.close(latency_figure)

        rps_axis.plot(
            seconds,
            [float(row["Requests/s"] or 0) for row in points],
            label=scenario,
        )

    rps_axis.set(
        title="Solicitudes /predict por segundo",
        xlabel="Tiempo transcurrido (s)",
        ylabel="Solicitudes/s",
    )
    rps_axis.legend(title="Escenario")
    rps_axis.grid(alpha=0.3)
    rps_figure.tight_layout()
    rps_figure.savefig(figures_dir / "rps_escenarios.png", dpi=150)
    plt.close(rps_figure)

    error_figure, error_axis = plt.subplots(figsize=(8, 5))
    error_axis.bar(
        [str(row["escenario"]) for row in summaries],
        [float(row["tasa_error"]) * 100 for row in summaries],
    )
    error_axis.set(
        title="Tasa de error de /predict",
        xlabel="Escenario",
        ylabel="Solicitudes fallidas (%)",
    )
    error_axis.grid(axis="y", alpha=0.3)
    error_figure.tight_layout()
    error_figure.savefig(figures_dir / "tasa_error.png", dpi=150)
    plt.close(error_figure)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Genera el resumen y las gráficas de carga de Locust."
    )
    parser.add_argument("--runs-dir", default="reports/load_test")
    parser.add_argument("--output-dir", default="reports/load_test")
    args = parser.parse_args()

    make_plots(args.runs_dir, args.output_dir)
    print(f"Reporte generado en {args.output_dir}")


if __name__ == "__main__":
    main()
