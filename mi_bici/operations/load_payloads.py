"""Entradas reproducibles para las pruebas de carga de la API P2."""

import math
import random


def make_payload(rng: random.Random) -> dict:
    """Genera las variables de una estación durante una hora."""
    capacity = rng.randint(10, 35)
    hour = rng.randint(0, 23)
    day = rng.randint(0, 6)

    departures = float(rng.randint(0, 8))
    arrivals = float(rng.randint(0, 8))
    net_flow = departures - arrivals

    # El primer elemento es la hora anterior; el último, hace 24 horas.
    history = [(float(rng.randint(0, 8)), float(rng.randint(0, 8))) for _ in range(24)]

    payload = {
        "departures": departures,
        "arrivals": arrivals,
        "net_flow": net_flow,
        "absolute_imbalance": abs(net_flow),
        "imbalance_ratio": abs(net_flow) / capacity,
        "capacity": capacity,
        "latitude": 20.6820,
        "longitude": -103.3500,
        "hour_of_day": hour,
        "day_of_week": day,
        "is_weekend": int(day >= 5),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * day / 7),
        "dow_cos": math.cos(2 * math.pi * day / 7),
    }

    for index, name in enumerate(("departures", "arrivals")):
        values = [record[index] for record in history]
        payload[f"{name}_lag_1h"] = values[0]
        payload[f"{name}_lag_2h"] = values[1]
        payload[f"{name}_lag_24h"] = values[23]
        payload[f"{name}_rolling_3h"] = sum(values[:3]) / 3
        payload[f"{name}_rolling_24h"] = sum(values) / 24

    net_history = [departure - arrival for departure, arrival in history]
    payload["net_flow_lag_1h"] = net_history[0]
    payload["net_flow_lag_2h"] = net_history[1]
    payload["net_flow_lag_24h"] = net_history[23]
    payload["net_flow_rolling_3h"] = sum(net_history[:3]) / 3
    payload["net_flow_rolling_24h"] = sum(net_history) / 24

    return payload


def make_invalid_payload(rng: random.Random) -> dict:
    """Genera una entrada que la API debe rechazar por hora inválida."""
    payload = make_payload(rng)
    payload["hour_of_day"] = 24
    return payload
