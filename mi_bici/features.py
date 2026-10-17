"""
Generación de características para P2: clasificación de riesgo operacional.

Pregunta de negocio de P2
-------------------------
¿Una estación presentará un nivel alto de desequilibrio operacional
durante la siguiente hora?

Momento de predicción
---------------------
El modelo utiliza la información disponible al finalizar la hora t para
predecir el desequilibrio operacional durante la hora t+1.

Limitación importante
---------------------
Los datos abiertos de MiBici no contienen el número real de bicicletas
disponibles ni el número de espacios libres en cada estación a lo largo
del tiempo.

Por esta razón, P2 no predice directamente una estación vacía o saturada.
En su lugar, predice un ALTO DESEQUILIBRIO OPERACIONAL como aproximación
(proxy) del riesgo operacional.

La medida utilizada es:

    imbalance_ratio = |departures - arrivals| / capacidad_estacion

La etiqueta binaria Risk=1/0 NO se calcula en este archivo. El umbral se
calcula posteriormente en train.py utilizando únicamente el conjunto de
entrenamiento para evitar fuga de información.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mi_bici.config import PROCESSED_DATA_DIR

app = typer.Typer()


def load_trips(processed_dir: Path) -> pd.DataFrame:
    """
    Carga todos los archivos Parquet mensuales de viajes de MiBici.

    Retorna
    -------
    pd.DataFrame
        Registros de viajes necesarios para construir las variables
        agregadas por estación y hora.

    Justificación para P2
    ---------------------
    P2 necesita utilizar el historial de viajes para determinar cuántos
    viajes salen y cuántos llegan a cada estación durante cada hora.

    Fuga de información
    -------------------
    Esta función únicamente carga información histórica de viajes. No
    incorpora información del resultado futuro que queremos predecir.
    """

    files = sorted(processed_dir.glob("viajes_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No se encontraron archivos viajes_*.parquet en {processed_dir}")

    frames = []

    for path in files:
        logger.info(f"Leyendo {path.name}")

        frames.append(
            pd.read_parquet(
                path,
                columns=[
                    "trip_id",
                    "started_at",
                    "ended_at",
                    "origin_station_id",
                    "destination_station_id",
                ],
            )
        )

    trips = pd.concat(
        frames,
        ignore_index=True,
    )

    trips["started_at"] = pd.to_datetime(
        trips["started_at"],
        errors="coerce",
    )

    trips["ended_at"] = pd.to_datetime(
        trips["ended_at"],
        errors="coerce",
    )

    trips = trips.dropna(
        subset=[
            "started_at",
            "ended_at",
            "origin_station_id",
            "destination_station_id",
        ]
    ).copy()

    # Se normalizan los identificadores para que puedan compararse
    # correctamente con los identificadores del catálogo de estaciones.
    for col in [
        "origin_station_id",
        "destination_station_id",
    ]:
        trips[col] = trips[col].astype("string").str.strip()

    logger.success(f"Viajes cargados: {len(trips):,}")

    return trips


def load_stations(processed_dir: Path) -> pd.DataFrame:
    """
    Carga el catálogo normalizado de estaciones de MiBici.

    Retorna
    -------
    pd.DataFrame
        Información de las estaciones, incluyendo capacidad y ubicación.

    Justificación para P2
    ---------------------
    La capacidad de la estación es necesaria para normalizar el
    desequilibrio operacional.

    Esto permite comparar estaciones de diferentes tamaños. Por ejemplo,
    un desequilibrio de 10 viajes representa una presión distinta en una
    estación con capacidad de 20 bicicletas que en una estación con
    capacidad de 80.

    Las coordenadas y el nombre de la estación se conservan como
    información contextual.

    Fuga de información
    -------------------
    La capacidad y ubicación son atributos de la estación y no contienen
    información sobre el comportamiento futuro de los viajes.
    """

    path = processed_dir / "estaciones.parquet"

    if not path.is_file():
        raise FileNotFoundError(f"No se encontró {path}")

    stations = pd.read_parquet(
        path,
        columns=[
            "station_id",
            "station_name",
            "capacity",
            "latitude",
            "longitude",
        ],
    )

    stations["station_id"] = stations["station_id"].astype("string").str.strip()

    stations["capacity"] = pd.to_numeric(
        stations["capacity"],
        errors="coerce",
    )

    # Se conserva una sola fila por estación.
    stations = stations.drop_duplicates(
        subset=["station_id"],
        keep="last",
    )

    # Una estación sin capacidad válida no puede utilizarse para normalizar
    # el desequilibrio.
    stations = stations[stations["capacity"] > 0].copy()

    logger.success(f"Estaciones válidas: {len(stations):,}")

    return stations


def aggregate_station_flows(
    trips: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrega los viajes por estación y hora.

    Variables generadas
    --------------------
    departures
        Número de viajes que salen de una estación durante una hora.

    arrivals
        Número de viajes que llegan a una estación durante una hora.

    Justificación para P2
    ---------------------
    Estas variables permiten responder:

        ¿Qué nivel de movimiento está experimentando actualmente
        la estación?

    Las salidas y llegadas son la información básica a partir de la cual
    se calcula el desequilibrio operacional.

    Fuga de información
    -------------------
    Cada viaje se asigna a la hora correspondiente a su timestamp
    observado. No se utilizan viajes de horas futuras para calcular
    el flujo de una hora determinada.
    """

    departures = (
        trips.assign(hour=trips["started_at"].dt.floor("h"))
        .groupby(
            [
                "origin_station_id",
                "hour",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "origin_station_id": "station_id",
                "size": "departures",
            }
        )
    )

    arrivals = (
        trips.assign(hour=trips["ended_at"].dt.floor("h"))
        .groupby(
            [
                "destination_station_id",
                "hour",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "destination_station_id": "station_id",
                "size": "arrivals",
            }
        )
    )

    return departures, arrivals


def create_station_hour_grid(
    departures: pd.DataFrame,
    arrivals: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye una observación para cada combinación estación-hora.

    Justificación para P2
    ---------------------
    Que una estación no aparezca en los viajes agregados durante una hora
    no significa necesariamente que falte información. Puede significar
    simplemente que no hubo viajes durante esa hora.

    Por eso se construye una cuadrícula completa de:

        estación × hora

    y posteriormente los valores faltantes de salidas y llegadas se
    interpretan como cero.

    Esta estructura también es necesaria para calcular correctamente los
    rezagos y las ventanas móviles.
    """

    min_hour = min(
        departures["hour"].min(),
        arrivals["hour"].min(),
    )

    max_hour = max(
        departures["hour"].max(),
        arrivals["hour"].max(),
    )

    hours = pd.date_range(
        min_hour,
        max_hour,
        freq="h",
    )

    full_grid = (
        stations[["station_id"]]
        .assign(key=1)
        .merge(
            pd.DataFrame(
                {
                    "hour": hours,
                    "key": 1,
                }
            ),
            on="key",
        )
        .drop(columns="key")
    )

    hourly = full_grid.merge(
        departures,
        on=[
            "station_id",
            "hour",
        ],
        how="left",
    )

    hourly = hourly.merge(
        arrivals,
        on=[
            "station_id",
            "hour",
        ],
        how="left",
    )

    # Si no hubo viajes durante una hora, se interpreta como cero actividad.
    hourly["departures"] = hourly["departures"].fillna(0)

    hourly["arrivals"] = hourly["arrivals"].fillna(0)

    # Se incorporan los atributos estáticos de la estación.
    hourly = hourly.merge(
        stations,
        on="station_id",
        how="left",
        validate="many_to_one",
    )

    return hourly


def add_imbalance_features(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula las variables relacionadas con el desequilibrio operacional.

    Variables generadas
    --------------------
    net_flow
        departures - arrivals.

    absolute_imbalance
        Valor absoluto de departures - arrivals.

    imbalance_ratio
        Desequilibrio absoluto normalizado por la capacidad de la estación.

    Justificación para P2
    ---------------------
    Estas variables permiten responder:

        ¿La estación está presentando presión hacia agotamiento
        o hacia saturación?

    Interpretación de net_flow:

        net_flow > 0
            Hay más salidas que llegadas.
            Existe presión hacia el agotamiento de bicicletas.

        net_flow < 0
            Hay más llegadas que salidas.
            Existe presión hacia la saturación de espacios.

        net_flow ≈ 0
            Las entradas y salidas están relativamente equilibradas.

    El valor absoluto mide la magnitud del desequilibrio.

    La normalización por capacidad permite comparar estaciones de
    diferentes tamaños.

    Fuga de información
    -------------------
    Estas variables utilizan únicamente la hora actual ya completada y
    la capacidad de la estación. No utilizan información de t+1.
    """

    hourly["net_flow"] = hourly["departures"] - hourly["arrivals"]

    hourly["absolute_imbalance"] = hourly["net_flow"].abs()

    hourly["imbalance_ratio"] = hourly["absolute_imbalance"] / hourly["capacity"]

    return hourly


def add_temporal_features(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera variables relacionadas con patrones temporales.

    Variables generadas
    --------------------
    hour_of_day
        Hora del día, de 0 a 23.

    day_of_week
        Día de la semana.

    is_weekend
        Indica si la observación pertenece al sábado o domingo.

    hour_sin / hour_cos
        Representación cíclica de la hora del día.

    dow_sin / dow_cos
        Representación cíclica del día de la semana.

    Justificación para P2
    ---------------------
    Estas variables permiten responder:

        ¿El comportamiento operacional cambia dependiendo de la hora
        o del día de la semana?

    Por ejemplo, una estación puede presentar comportamientos distintos
    durante las horas de entrada y salida laboral.

    Las variables seno y coseno permiten representar la naturaleza cíclica
    del tiempo. Por ejemplo, las horas 23 y 0 son consecutivas aunque sus
    valores numéricos sean muy diferentes.

    Fuga de información
    -------------------
    Estas variables se obtienen únicamente del timestamp de la observación.
    No utilizan información sobre viajes futuros.
    """

    hourly["hour_of_day"] = hourly["hour"].dt.hour

    hourly["day_of_week"] = hourly["hour"].dt.dayofweek

    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)

    hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour_of_day"] / 24)

    hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour_of_day"] / 24)

    hourly["dow_sin"] = np.sin(2 * np.pi * hourly["day_of_week"] / 7)

    hourly["dow_cos"] = np.cos(2 * np.pi * hourly["day_of_week"] / 7)

    return hourly


def add_historical_features(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera variables históricas mediante rezagos y ventanas móviles.

    Variables generadas para departures, arrivals y net_flow
    ----------------------------------------------------------
    - lag_1h
    - lag_2h
    - lag_24h
    - rolling_3h
    - rolling_24h

    Justificación para P2
    ---------------------
    Estas variables permiten responder:

        ¿El comportamiento reciente de una estación ayuda a explicar
        su comportamiento durante la siguiente hora?

    Los rezagos representan valores específicos del pasado.

    Las ventanas móviles representan el comportamiento promedio reciente
    y permiten identificar tendencias sin depender de una sola hora.

    Prevención de fuga de información
    ---------------------------------
    Antes de calcular cada ventana móvil se utiliza shift(1).

    Por ejemplo:

        rolling_3h(t)

    utiliza únicamente:

        t-3, t-2 y t-1

    y no utiliza:

        t ni t+1.

    De esta manera, las variables históricas contienen únicamente
    información disponible antes de realizar la predicción.
    """

    hourly = hourly.sort_values(
        [
            "station_id",
            "hour",
        ]
    ).reset_index(drop=True)

    grouped = hourly.groupby(
        "station_id",
        group_keys=False,
    )

    for column in [
        "departures",
        "arrivals",
        "net_flow",
    ]:
        hourly[f"{column}_lag_1h"] = grouped[column].shift(1)

        hourly[f"{column}_lag_2h"] = grouped[column].shift(2)

        hourly[f"{column}_lag_24h"] = grouped[column].shift(24)

        hourly[f"{column}_rolling_3h"] = grouped[column].transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    3,
                    min_periods=1,
                )
                .mean()
            )
        )

        hourly[f"{column}_rolling_24h"] = grouped[column].transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    24,
                    min_periods=1,
                )
                .mean()
            )
        )

    return hourly


def add_next_hour_target(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera las variables objetivo correspondientes a la siguiente hora.

    Variables generadas
    --------------------
    target_departures
        Salidas durante la hora t+1.

    target_arrivals
        Llegadas durante la hora t+1.

    target_net_flow
        Flujo neto durante la hora t+1.

    target_imbalance_ratio
        Desequilibrio normalizado durante la hora t+1.

    Justificación para P2
    ---------------------
    Estas variables representan exactamente el comportamiento que queremos
    predecir:

        Información disponible hasta t
                    ↓
                 modelo
                    ↓
        desequilibrio durante t+1

    Prevención de fuga de información
    ---------------------------------
    El target utiliza información de t+1 porque esa es precisamente la
    variable que queremos predecir.

    Sin embargo, estas variables objetivo NO se utilizan como variables
    de entrada al modelo.

    La etiqueta binaria Risk=1/0 tampoco se genera aquí. El umbral se
    calcula posteriormente en train.py utilizando únicamente el conjunto
    de entrenamiento.
    """

    hourly = hourly.sort_values(
        [
            "station_id",
            "hour",
        ]
    ).reset_index(drop=True)

    grouped = hourly.groupby(
        "station_id",
        group_keys=False,
    )

    hourly["target_departures"] = grouped["departures"].shift(-1)

    hourly["target_arrivals"] = grouped["arrivals"].shift(-1)

    hourly["target_net_flow"] = hourly["target_departures"] - hourly["target_arrivals"]

    hourly["target_imbalance_ratio"] = hourly["target_net_flow"].abs() / hourly["capacity"]

    # La última observación de cada estación no tiene información de t+1.
    hourly = hourly.dropna(subset=["target_imbalance_ratio"]).copy()

    return hourly


def build_station_hour_table(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el conjunto completo de características de P2.

    Flujo del proceso
    -----------------
    1. Agregar salidas y llegadas por estación-hora.
    2. Crear la cuadrícula completa estación-hora.
    3. Calcular variables de desequilibrio.
    4. Calcular variables temporales.
    5. Calcular variables históricas.
    6. Crear el objetivo de la siguiente hora.

    Esta función coordina las funciones individuales de generación de
    características para mantener el proceso modular y reutilizable.
    """

    departures, arrivals = aggregate_station_flows(trips)

    hourly = create_station_hour_grid(
        departures=departures,
        arrivals=arrivals,
        stations=stations,
    )

    hourly = add_imbalance_features(hourly)

    hourly = add_temporal_features(hourly)

    hourly = add_historical_features(hourly)

    hourly = add_next_hour_target(hourly)

    # Se convierten infinitos a valores faltantes para evitar problemas
    # posteriores durante el entrenamiento del modelo.
    hourly = hourly.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return hourly


@app.command()
def main(
    output_path: Path = (PROCESSED_DATA_DIR / "features_p2.parquet"),
):
    """
    Genera el dataset final de características para P2.

    Unidad de análisis:
        estación-hora

    Objetivo:
        predecir el desequilibrio operacional de la siguiente hora.

    La etiqueta binaria Risk=1/0 se calcula posteriormente en train.py
    utilizando un umbral calculado únicamente con el conjunto de
    entrenamiento.
    """

    logger.info("Generando features para P2...")

    trips = load_trips(PROCESSED_DATA_DIR)

    stations = load_stations(PROCESSED_DATA_DIR)

    features = build_station_hour_table(
        trips,
        stations,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    logger.success(f"Features P2 generadas: {output_path.name} ({len(features):,} observaciones)")

    logger.info(
        "Risk=1/0 se calculará en train.py con el percentil 80 calculado únicamente sobre train."
    )


if __name__ == "__main__":
    app()
