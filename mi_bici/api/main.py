"""Aplicación FastAPI: ``GET /health`` (L3) y ``POST /predict`` (M3).

Se arranca con ``uvicorn mi_bici.api.main:app``. El endpoint de predicción usa un
``Predictor`` (``mi_bici/api/predictor.py``); no carga el modelo directamente.
"""
