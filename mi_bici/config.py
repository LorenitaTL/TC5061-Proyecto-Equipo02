import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproducibility
RANDOM_STATE = 42

# MLflow
TEAM = "equipo02"
MLFLOW_EXPERIMENT_NAME = f"mibici-{TEAM}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlartifacts"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# MiBici: configuración de ingesta
MIBICI_BASE_URL = "https://mibici.net"

DATA_START_MONTH = "2025-01"
DATA_END_MONTH = "2026-06"
TRAIN_END_MONTH = "2025-12"

STATIONS_FILENAME = "nomenclatura_2026_09.csv"

REFERENCES_DIR = PROJ_ROOT / "references"

# ---------------------------------------------------------
# Configuración del modelo P1
# ---------------------------------------------------------

# Fechas de corte (exclusivas) de la partición cronológica.
P1_TRAIN_END = "2026-04-01"
P1_VALIDATION_END = "2026-06-01"

P1_SAMPLE_SIZE = 1_000_000

# ---------------------------------------------------------
# Configuración del modelo P2
# ---------------------------------------------------------

# La semilla RANDOM_STATE se define en la sección "Reproducibility".

# División temporal de los datos.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

# Percentil utilizado para definir el umbral de alto riesgo.
RISK_PERCENTILE = 0.80

# Hiperparámetros de Logistic Regression.
LOGISTIC_MAX_ITER = 1000
LOGISTIC_CLASS_WEIGHT = "balanced"
LOGISTIC_SOLVER = "lbfgs"

# Estrategia de imputación de valores faltantes.

IMPUTER_STRATEGY = "median"
