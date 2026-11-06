import pandas as pd

from mi_bici.api.model_loader import load_p2_model
from mi_bici.api.predictor import FEATURE_COLUMNS
from mi_bici.config import PROCESSED_DATA_DIR


def test_p2_model_serving_real_inference():
    """Verifica que el modelo champion de P2 pueda cargarse y predecir."""

    # Datos reales de P2
    path = PROCESSED_DATA_DIR / "features_p2.parquet"

    df = pd.read_parquet(path, columns=FEATURE_COLUMNS).dropna().head(1)

    # Cargar modelo desde MLflow Model Registry
    predictor = load_p2_model()

    # Ejecutar inferencia
    predictions = predictor.predict(df)

    # Validaciones del contrato
    assert len(predictions) == 1

    prediction = predictions[0]

    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.risk_class in (0, 1)

    # Verificar que el threshold utilizado sea el registrado
    assert predictor.threshold == 0.35

    print()
    print("=== E4 MODEL SERVING ===")
    print(f"MODEL: {predictor.model_name}")
    print(f"VERSION: {predictor.model_version}")
    print(f"THRESHOLD: {predictor.threshold}")
    print(f"PROBABILITY: {prediction.probability}")
    print(f"RISK_CLASS: {prediction.risk_class}")
