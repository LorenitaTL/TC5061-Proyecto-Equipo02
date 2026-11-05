"""Carga del modelo de P2 desde el Model Registry de MLflow (E1 y E2).

``main.py`` llama a ``load_predictor(settings: Settings) -> Predictor`` al arrancar la API.
Debe devolver un objeto que cumpla ``Predictor`` (``mi_bici/api/predictor.py``): apunta
MLflow a ``settings.mlflow_tracking_uri``, carga ``settings.model_uri``
(``models:/<MODEL_NAME>@<MODEL_ALIAS>`` o la versión fija), lee el umbral
``classification_threshold`` del run de esa versión y usa ``check_feature_columns`` antes
de predecir. Si algo falla, debe lanzar la excepción: la API arranca igual, sin modelo.
"""
