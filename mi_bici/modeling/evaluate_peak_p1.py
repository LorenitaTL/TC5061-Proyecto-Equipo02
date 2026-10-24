"""Evalúa horas pico de P1 y conserva la evidencia en MLflow."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from mlflow import MlflowClient
import mlflow.sklearn
import pandas as pd

from mi_bici.features_p1 import DATASET_PATH, FEATURES
from mi_bici.modeling.baseline_p1 import seasonal_prediction
from mi_bici.modeling.evaluate_p1 import predict_demand, regression_metrics
from mi_bici.modeling.splits_p1 import temporal_masks
from mi_bici.modeling.tracking import configure_mlflow, get_data_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--candidate-run-ids", nargs=3, required=True)
    parser.add_argument("--top-hours", type=int, default=4)
    args = parser.parse_args()

    if not 1 <= args.top_hours <= 24:
        raise ValueError("top-hours debe estar entre 1 y 24.")

    configure_mlflow()
    client = MlflowClient()
    run_ids = {
        "Baseline": args.baseline_run_id,
        **dict(zip(("A", "B", "C"), args.candidate_run_ids, strict=True)),
    }

    if len(set(run_ids.values())) != 4:
        raise ValueError("Se requieren cuatro Run ID diferentes.")

    version = get_data_version()
    saved_runs = {}

    # Verificar las corridas antes de añadir evidencia.
    for label, run_id in run_ids.items():
        saved = client.get_run(run_id)
        if saved.info.status != "FINISHED":
            raise ValueError(f"{label}: la corrida no está finalizada.")
        if saved.data.tags.get("question") != "P1":
            raise ValueError(f"{label}: la corrida no corresponde a P1.")
        if saved.data.tags.get("data_version") != version:
            raise ValueError(f"{label}: la versión de datos no coincide.")
        saved_runs[label] = saved

    df = pd.read_parquet(
        DATASET_PATH,
        columns=["hour", "trip_count", *FEATURES],
    )
    masks = temporal_masks(df["hour"])
    train = df.loc[masks["train"]]
    validation = df.loc[masks["validation"]]

    if train.empty or validation.empty:
        raise ValueError("Entrenamiento y validación deben contener datos.")

    hourly = (
        train.groupby(train["hour"].dt.hour)["trip_count"]
        .mean()
        .rename("mean_trip_count_train")
        .rename_axis("hour_of_day")
        .reset_index()
        .sort_values(
            ["mean_trip_count_train", "hour_of_day"],
            ascending=[False, True],
        )
    )
    selected_hours = hourly.head(args.top_hours)["hour_of_day"].tolist()
    hourly["selected_as_peak"] = hourly["hour_of_day"].isin(selected_hours)

    peak = validation.loc[validation["hour"].dt.hour.isin(selected_hours)]
    if peak.empty:
        raise ValueError("No hay observaciones de validación en horas pico.")

    peak_params = {
        "peak_hours": json.dumps(sorted(selected_hours)),
        "peak_top_hours": str(args.top_hours),
        "peak_definition": "top_mean_trip_count_by_clock_hour_in_training",
    }

    # Evitar cambiar parámetros ya registrados con otro significado.
    for label, saved in saved_runs.items():
        for key, value in peak_params.items():
            previous = saved.data.params.get(key)
            if previous is not None and previous != value:
                raise ValueError(f"{label}: conflicto con el parámetro {key}.")

    rows = []
    for label, run_id in run_ids.items():
        if label == "Baseline":
            predictions = seasonal_prediction(peak)
        else:
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo_p1")
            predictions = predict_demand(model, peak)

        rows.append(
            {
                "model": label,
                "run_id": run_id,
                "n_validation_peak": len(peak),
                **regression_metrics(peak["trip_count"], predictions, "peak"),
            }
        )

    comparison = pd.DataFrame(rows)
    for metric in ("mae_peak", "rmse_peak"):
        baseline = comparison.loc[comparison["model"] == "Baseline", metric].iloc[0]
        comparison[f"{metric}_improvement_pct"] = 100 * (baseline - comparison[metric]) / baseline

    configuration = {
        "data_version": version,
        "dataset": DATASET_PATH.name,
        "definition": peak_params["peak_definition"],
        "top_hours": args.top_hours,
        "selected_hours": sorted(selected_hours),
        "tie_break": "ascending hour_of_day",
        "selection_partition": "train",
        "evaluation_partition": "validation",
        "n_training": len(train),
        "n_validation": len(validation),
        "n_validation_peak": len(peak),
        "run_ids": run_ids,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    with TemporaryDirectory() as directory:
        folder = Path(directory)
        comparison.to_csv(folder / "peak_comparison.csv", index=False)
        hourly.to_csv(folder / "training_hourly_profile.csv", index=False)
        config_path = folder / "peak_evaluation.json"
        config_path.write_text(
            json.dumps(configuration, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for row in rows:
            run_id = row["run_id"]
            for key, value in peak_params.items():
                client.log_param(run_id, key, value)

            client.log_metric(run_id, "mae_peak", row["mae_peak"])
            client.log_metric(run_id, "rmse_peak", row["rmse_peak"])
            client.log_metric(run_id, "n_validation_peak", row["n_validation_peak"])

            for filename in (
                "peak_comparison.csv",
                "training_hourly_profile.csv",
            ):
                client.log_artifact(
                    run_id,
                    str(folder / filename),
                    artifact_path="data/peak_evaluation",
                )

            client.log_artifact(
                run_id,
                str(config_path),
                artifact_path="config/peak_evaluation",
            )
            client.log_artifact(
                run_id,
                str(Path(__file__).resolve()),
                artifact_path="config/peak_evaluation",
            )

    print("Horas pico:", sorted(selected_hours))
    print(comparison.round(6).to_string(index=False))
    print("OK: métricas y evidencia guardadas en las cuatro corridas.")


if __name__ == "__main__":
    main()
