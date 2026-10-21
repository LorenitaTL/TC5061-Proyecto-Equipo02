"""Gráficas de validación para los modelos candidatos de P1."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np


def log_validation_plots(y_true, y_pred) -> None:
    """Registra gráficas de validación en la corrida activa.

    Utiliza hasta 5,000 observaciones para mantener legibles las gráficas.
    Las métricas deben calcularse por separado sobre toda la validación.
    """
    if mlflow.active_run() is None:
        raise RuntimeError("Se requiere una corrida activa de MLflow.")

    real = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)

    if real.size == 0 or real.size != predicted.size:
        raise ValueError("Los valores reales y predichos deben tener igual longitud no vacía.")

    if not np.isfinite(real).all() or not np.isfinite(predicted).all():
        raise ValueError("Los valores reales y predichos deben ser finitos.")

    indices = np.linspace(
        0,
        real.size - 1,
        num=min(5_000, real.size),
        dtype=int,
    )
    real = real[indices]
    predicted = predicted[indices]
    residuals = real - predicted

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        ax.scatter(real, predicted, alpha=0.2, s=10)
        upper = max(float(real.max()), float(predicted.max()), 1.0)
        ax.plot([0, upper], [0, upper], linestyle="--", color="black")
        ax.set(
            xlabel="Viajes reales",
            ylabel="Viajes predichos",
            title="Validación: valores reales frente a predichos",
        )
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/validation_actual_vs_predicted.png")
    finally:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        ax.scatter(predicted, residuals, alpha=0.2, s=10)
        ax.axhline(0, linestyle="--", color="black")
        ax.set(
            xlabel="Viajes predichos",
            ylabel="Residual: real - predicho",
            title="Validación: residuales frente a predicciones",
        )
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/validation_residuals.png")
    finally:
        plt.close(fig)