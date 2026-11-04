"""Pruebas de la interfaz entre la API y el modelo (``mi_bici/api/predictor.py``)."""

import pandas as pd
import pytest

from mi_bici.api.predictor import (
    FEATURE_COLUMNS,
    FixedPredictor,
    Prediction,
    Predictor,
    check_feature_columns,
)


def feature_frame(rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)] * rows, columns=FEATURE_COLUMNS)


def test_feature_columns_son_las_del_entrenamiento_de_p2():
    from mi_bici.modeling.train_p2 import FEATURE_COLUMNS as train_columns

    assert FEATURE_COLUMNS == train_columns
    assert len(FEATURE_COLUMNS) == 30


def test_check_feature_columns_acepta_el_orden_del_entrenamiento():
    check_feature_columns(feature_frame())


def test_check_feature_columns_rechaza_otro_orden():
    df = feature_frame()[list(reversed(FEATURE_COLUMNS))]
    with pytest.raises(ValueError, match="el orden es distinto"):
        check_feature_columns(df)


def test_check_feature_columns_reporta_faltantes_y_sobrantes():
    df = feature_frame().drop(columns="capacity").assign(station_id="10")
    with pytest.raises(ValueError, match=r"faltan: \['capacity'\], sobran: \['station_id'\]"):
        check_feature_columns(df)


def test_fixed_predictor_devuelve_una_prediccion_por_fila():
    predictor: Predictor = FixedPredictor(probability=0.8, threshold=0.35)

    assert predictor.predict(feature_frame(rows=3)) == [Prediction(0.8, 1)] * 3


def test_fixed_predictor_aplica_el_umbral():
    assert FixedPredictor(probability=0.2).predict(feature_frame(1)) == [Prediction(0.2, 0)]
    assert FixedPredictor(probability=0.35).predict(feature_frame(1)) == [Prediction(0.35, 1)]


def test_fixed_predictor_valida_las_columnas():
    with pytest.raises(ValueError):
        FixedPredictor().predict(feature_frame().drop(columns="capacity"))
