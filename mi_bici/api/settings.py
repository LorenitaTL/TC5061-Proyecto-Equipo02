"""Configuración de la API por variables de entorno (L3).

| Variable | Valor por omisión | Uso |
| :--- | :--- | :--- |
| ``MLFLOW_TRACKING_URI`` | el de ``mi_bici/config.py`` (``mlflow.db`` del proyecto) | Servidor o base de MLflow |
| ``MODEL_NAME`` | ``mibici-equipo02-p2-riesgo`` | Modelo registrado de P2 |
| ``MODEL_ALIAS`` | ``champion`` | Alias que se resuelve si no hay versión |
| ``MODEL_VERSION`` | (vacío) | Versión fija; si se define, tiene prioridad sobre el alias |

Ningún valor se lee al importar el módulo: ``Settings.from_env()`` se llama al arrancar la
app, así que las pruebas y el contenedor pueden cambiar las variables sin recargar nada.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import os

from mi_bici.config import MLFLOW_TRACKING_URI
from mi_bici.modeling.register_p2 import ALIAS, MODEL_NAME


@dataclass(frozen=True)
class Settings:
    """Qué modelo sirve la API y dónde buscarlo."""

    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI
    model_name: str = MODEL_NAME
    model_alias: str = ALIAS
    model_version: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Lee la configuración de ``environ`` (por omisión, ``os.environ``).

        Una variable vacía cuenta como no definida, para que ``MODEL_VERSION=`` en un
        ``.env`` o en ``podman run -e`` no fije una versión vacía.
        """

        env = os.environ if environ is None else environ

        def value(name: str) -> str | None:
            return env.get(name, "").strip() or None

        return cls(
            mlflow_tracking_uri=value("MLFLOW_TRACKING_URI") or cls.mlflow_tracking_uri,
            model_name=value("MODEL_NAME") or cls.model_name,
            model_alias=value("MODEL_ALIAS") or cls.model_alias,
            model_version=value("MODEL_VERSION"),
        )

    @property
    def model_uri(self) -> str:
        """URI del Model Registry: por versión si está fijada, si no por alias."""

        if self.model_version:
            return f"models:/{self.model_name}/{self.model_version}"
        return f"models:/{self.model_name}@{self.model_alias}"
