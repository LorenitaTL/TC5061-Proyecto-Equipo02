from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from mi_bici.config import DATA_START_MONTH, PROCESSED_DATA_DIR, TRAIN_END_MONTH

app = typer.Typer()

# P3: perfiles de estaciones
PROFILE_TRIP_COLUMNS = [
    "started_at",
    "ended_at",
    "origin_station_id",
    "destination_station_id",
    "zero_duration",
]
AM_PEAK_HOURS = range(6, 10)
PM_PEAK_HOURS = range(17, 21)
MIN_DEPARTURES = 1000


def load_profile_trips(
    start_month: str = DATA_START_MONTH,
    end_month: str = TRAIN_END_MONTH,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Carga los viajes normalizados de un rango de meses para construir perfiles de P3.

    Por defecto usa solo el periodo de entrenamiento (DATA_START_MONTH a TRAIN_END_MONTH),
    para que los perfiles no incorporen información de los meses de prueba.
    """
    months = pd.period_range(start_month, end_month, freq="M").strftime("%Y_%m")
    paths = [processed_dir / f"viajes_{month}.parquet" for month in months]

    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"No se encontraron los archivos de viajes: {missing}")

    trips = pd.concat(
        [pd.read_parquet(path, columns=PROFILE_TRIP_COLUMNS) for path in paths],
        ignore_index=True,
    )

    logger.success(f"Viajes cargados para perfiles ({start_month} a {end_month}): {len(trips):,}")

    return trips


def flow_balance(trips: pd.DataFrame, hours: range = range(24)) -> pd.Series:
    """Calcula (llegadas - salidas) / (llegadas + salidas) por estación en una franja horaria.

    Es positivo en estaciones destino, que acumulan bicis, y negativo en estaciones origen,
    que se vacían. En la hora pico de la mañana separa estaciones residenciales (negativo)
    de destinos de trabajo o escuela (positivo), que es la diferencia principal entre
    perfiles de uso en P3.
    """
    departures = (
        trips.loc[trips["started_at"].dt.hour.isin(hours)].groupby("origin_station_id").size()
    )
    arrivals = (
        trips.loc[trips["ended_at"].dt.hour.isin(hours)].groupby("destination_station_id").size()
    )
    departures, arrivals = departures.align(arrivals, fill_value=0)

    return ((arrivals - departures) / (arrivals + departures)).rename_axis("station_id")


def build_station_profiles(
    trips: pd.DataFrame,
    min_departures: int = MIN_DEPARTURES,
) -> pd.DataFrame:
    """Construye una fila por estación con las variables que describen su perfil de uso (P3).

    Variables:

    - ``balance_am`` y ``balance_pm``: balance de flujo en las horas pico de la mañana
      (6-9 h) y de la tarde (17-20 h) en días laborales. Distinguen estaciones
      residenciales, destinos laborales y mixtas.
    - ``balance_dia``: balance de todo el día laboral. Identifica estaciones que pierden o
      acumulan bicis de forma sistemática, que es lo que define la necesidad de
      redistribución.
    - ``share_fin_semana``: proporción de salidas en sábado o domingo. Separa estaciones de
      uso recreativo, cuyo patrón no depende del commuting.
    - ``log_viajes_dia``: log(1 + salidas por día). El volumen se usa para describir los
      clusters, no para formarlos, para que no dominen las estaciones más usadas.

    Se descartan los viajes de duración cero, que no representan uso real, y las estaciones
    con menos de ``min_departures`` salidas, que solo operaron parte del periodo y no tienen
    un perfil completo.
    """
    valid = trips.loc[~trips["zero_duration"]]

    departures = valid.groupby("origin_station_id").size().rename_axis("station_id")
    active = departures.index[departures >= min_departures]

    is_weekday = valid["started_at"].dt.dayofweek < 5
    weekday = valid.loc[is_weekday]
    weekend_departures = (
        valid.loc[~is_weekday]
        .groupby("origin_station_id")
        .size()
        .rename_axis("station_id")
        .reindex(departures.index, fill_value=0)
    )
    n_days = valid["started_at"].dt.normalize().nunique()

    profiles = pd.DataFrame(
        {
            "balance_am": flow_balance(weekday, AM_PEAK_HOURS),
            "balance_pm": flow_balance(weekday, PM_PEAK_HOURS),
            "balance_dia": flow_balance(weekday),
            "share_fin_semana": weekend_departures / departures,
            "log_viajes_dia": np.log1p(departures / n_days),
        }
    )
    profiles = profiles.loc[profiles.index.isin(active)].dropna().sort_index()

    excluded = len(departures) - len(profiles)
    logger.success(f"Perfiles construidos: {len(profiles):,} estaciones ({excluded} excluidas)")

    return profiles


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
