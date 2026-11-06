"""Esquemas Pydantic de entrada y salida de POST /predict."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonNegativeFloat = Annotated[float, Field(ge=0)]
CyclicFeature = Annotated[float, Field(ge=-1, le=1)]


class PredictionRequest(BaseModel):
    """Variables de una estación y hora requeridas por el modelo P2."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "departures": 4.0,
                    "arrivals": 3.0,
                    "net_flow": 1.0,
                    "absolute_imbalance": 1.0,
                    "imbalance_ratio": 0.0667,
                    "capacity": 15,
                    "latitude": 20.6820,
                    "longitude": -103.3500,
                    "hour_of_day": 12,
                    "day_of_week": 3,
                    "is_weekend": 0,
                    "hour_sin": 0.0,
                    "hour_cos": -1.0,
                    "dow_sin": 0.4339,
                    "dow_cos": -0.9010,
                    "departures_lag_1h": 3.0,
                    "departures_lag_2h": 2.0,
                    "departures_lag_24h": 5.0,
                    "departures_rolling_3h": 2.3333,
                    "departures_rolling_24h": 1.9167,
                    "arrivals_lag_1h": 3.0,
                    "arrivals_lag_2h": 1.0,
                    "arrivals_lag_24h": 3.0,
                    "arrivals_rolling_3h": 1.6667,
                    "arrivals_rolling_24h": 2.1667,
                    "net_flow_lag_1h": 0.0,
                    "net_flow_lag_2h": 1.0,
                    "net_flow_lag_24h": 2.0,
                    "net_flow_rolling_3h": 0.6667,
                    "net_flow_rolling_24h": -0.25,
                }
            ]
        },
    )

    departures: NonNegativeFloat
    arrivals: NonNegativeFloat
    net_flow: float
    absolute_imbalance: NonNegativeFloat
    imbalance_ratio: NonNegativeFloat
    capacity: Annotated[int, Field(gt=0)]
    latitude: Annotated[float, Field(ge=-90, le=90)]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    hour_of_day: Annotated[int, Field(ge=0, le=23)]
    day_of_week: Annotated[int, Field(ge=0, le=6)]
    is_weekend: Annotated[int, Field(ge=0, le=1)]
    hour_sin: CyclicFeature
    hour_cos: CyclicFeature
    dow_sin: CyclicFeature
    dow_cos: CyclicFeature
    departures_lag_1h: NonNegativeFloat
    departures_lag_2h: NonNegativeFloat
    departures_lag_24h: NonNegativeFloat
    departures_rolling_3h: NonNegativeFloat
    departures_rolling_24h: NonNegativeFloat
    arrivals_lag_1h: NonNegativeFloat
    arrivals_lag_2h: NonNegativeFloat
    arrivals_lag_24h: NonNegativeFloat
    arrivals_rolling_3h: NonNegativeFloat
    arrivals_rolling_24h: NonNegativeFloat
    net_flow_lag_1h: float
    net_flow_lag_2h: float
    net_flow_lag_24h: float
    net_flow_rolling_3h: float
    net_flow_rolling_24h: float
