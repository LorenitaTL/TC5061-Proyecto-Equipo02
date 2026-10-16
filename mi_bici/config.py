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

STATIONS_FILENAME = "nomenclatura_2026_09.csv"

REFERENCES_DIR = PROJ_ROOT / "references"

# ---------------------------------------------------------
# Configuración del modelo P2
# ---------------------------------------------------------

# Semilla utilizada para garantizar reproducibilidad.
RANDOM_STATE = 42

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
