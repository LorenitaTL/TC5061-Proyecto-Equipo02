from mi_bici.api.model_loader import build_features, load_p2_model
from mi_bici.api.predictor import FEATURE_COLUMNS


def test_p2_model_serving_real_inference():
    """Verifica el serving del modelo P2 con un caso sintético."""

    # Caso de entrada sintético con las 30 features esperadas por P2.
    data = {column: 0.0 for column in FEATURE_COLUMNS}

    # Valores mínimos coherentes para algunas variables.
    data.update(
        {
            "departures": 5.0,
            "arrivals": 3.0,
            "net_flow": 2.0,
            "absolute_imbalance": 2.0,
            "imbalance_ratio": 0.25,
            "capacity": 20.0,
            "latitude": 19.4326,
            "longitude": -99.1332,
            "hour_of_day": 12.0,
            "day_of_week": 2.0,
            "is_weekend": 0.0,
            "hour_sin": 0.0,
            "hour_cos": -1.0,
            "dow_sin": 0.9749,
            "dow_cos": -0.2225,
        }
    )

    # E2: construir el DataFrame respetando exactamente el contrato de features.
    features = build_features(data)

    assert features.shape == (1, len(FEATURE_COLUMNS))
    assert list(features.columns) == FEATURE_COLUMNS

    # E1: cargar el modelo P2 desde MLflow Model Registry.
    predictor = load_p2_model()

    # E3: ejecutar inferencia real.
    predictions = predictor.predict(features)

    # Validaciones del contrato de serving.
    assert len(predictions) == 1

    prediction = predictions[0]

    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.risk_class in (0, 1)

    # El threshold debe corresponder al registrado para el modelo champion.
    assert predictor.threshold == 0.35

    print()
    print("=== E4 MODEL SERVING ===")
    print(f"MODEL: {predictor.model_name}")
    print(f"VERSION: {predictor.model_version}")
    print(f"THRESHOLD: {predictor.threshold}")
    print(f"PROBABILITY: {prediction.probability}")
    print(f"RISK_CLASS: {prediction.risk_class}")
