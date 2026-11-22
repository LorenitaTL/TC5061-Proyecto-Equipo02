"""Escenario de carga para la API P2 de MiBici."""

import random

from locust import HttpUser, between, task

from mi_bici.operations.load_payloads import make_invalid_payload, make_payload


class ApiUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        self.rng = random.Random()

    @task(8)
    def predict(self):
        with self.client.post(
            "/predict",
            json=make_payload(self.rng),
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Predicción: se esperaba 200, llegó {response.status_code}")

    @task(1)
    def health(self):
        with self.client.get("/health", name="/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Health: se esperaba 200, llegó {response.status_code}")
            elif not response.json().get("model_loaded"):
                response.failure("La API respondió, pero no tiene el modelo cargado")

    @task(1)
    def predict_invalido(self):
        with self.client.post(
            "/predict",
            json=make_invalid_payload(self.rng),
            name="/predict [422 esperado]",
            catch_response=True,
        ) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"Validación: se esperaba 422, llegó {response.status_code}")
