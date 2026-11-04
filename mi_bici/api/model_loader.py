"""Carga del modelo de P2 desde el Model Registry de MLflow (E1 y E2).

Debe devolver un objeto que cumpla ``Predictor`` (``mi_bici/api/predictor.py``): resuelve
``models:/<MODEL_NAME>@<MODEL_ALIAS>``, lee el umbral ``classification_threshold`` del run
de esa versión y usa ``check_feature_columns`` antes de predecir.
"""
