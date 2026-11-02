# P2 — Resultados

Este documento resume los resultados vigentes de P2 después de las correcciones del tutor de
la Actividad 7. La versión de la Entrega 2, que usaba la regresión logística como baseline, una
partición por proporciones 70/15/15 y la etiqueta con `>=`, queda en el historial de Git.

Cambios respecto a la Entrega 2:

| Comentario del tutor | Cambio |
|---|---|
| El baseline era una regresión logística | El baseline oficial es la regla de persistencia; la regresión logística es una candidata más. |
| El umbral dejaba 28 % de positivos | La etiqueta usa `>` en lugar de `>=`; con eso train queda en 19.40 % de positivos. |
| Comparar familias de modelos distintas | Se comparan regresión logística, HistGradientBoosting y Random Forest. |
| El modelo elegido no superaba al baseline | HistGradientBoosting supera a la persistencia en validation y test. |
| Unificar la partición de P1 y P2 | P2 usa las mismas fechas de corte que P1. |
| Linaje incompleto en el Registry | `register_p2` exige `git_commit` y `data_version`. |

---

## Objetivo

Predecir si una estación tendrá un desbalance operacional alto en la hora siguiente. Para cada
estación y hora `t`, el modelo usa la información disponible hasta `t` para predecir si el
desbalance relativo de `t+1` estará en el 20 % superior de la distribución de train.

## Partición temporal

La partición usa fechas de corte fijas, las mismas de P1, definidas en `mi_bici/config.py`
(`P2_TRAIN_END` y `P2_VALIDATION_END`) y aplicadas en `mi_bici/modeling/splits_p2.py`:

| Partición | Periodo | Observaciones | Positivos |
|---|---|---:|---:|
| Train | 2025-01-01 → 2026-03-31 | 5,241,600 | 19.40 % |
| Validation | 2026-04-01 → 2026-05-31 | 702,720 | 17.65 % |
| Test | 2026-06-01 → 2026-06-30 | 345,600 | 17.31 % |

Validation se usa para elegir la familia, los hiperparámetros y el threshold de clasificación.
Test se reserva para la evaluación final.

## Definición del riesgo

El umbral es el percentil 80 de `target_imbalance_ratio`, calculado solo sobre train y aplicado
sin cambios a validation y test:

```text
risk_threshold = 0.066667

Risk = 1 si target_imbalance_ratio > risk_threshold
Risk = 0 en otro caso
```

Hay muchas observaciones empatadas exactamente en el P80. Con `>=`, esos empates entraban a la
clase positiva y train quedaba con 28.33 % de positivos en lugar de cerca de 20 %.

## Baseline de persistencia

El baseline predice que la hora siguiente será de riesgo si el desbalance de la hora actual ya
supera el umbral:

```text
Predicción Risk = 1 si imbalance_ratio > risk_threshold
```

Usa el mismo umbral, la misma comparación `>` y la misma partición que las candidatas, y no
entrena nada. Se registra como `p2-baseline-persistencia-p80`, con `stage_of_work=baseline` y
`model_family=persistence`.

## Comparación de familias (validation)

Para cada familia se eligió el threshold de clasificación con mejor F1 entre los que alcanzan
una precisión mínima de 0.50:

| Modelo | Threshold | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Persistencia (baseline) | — | 0.4468 | 0.4469 | 0.4468 | 0.7288 |
| Logistic Regression | 0.70 | 0.5181 | 0.4395 | 0.4756 | 0.8334 |
| HistGradientBoosting | 0.35 | 0.5158 | 0.6910 | 0.5907 | 0.8882 |
| Random Forest | 0.55 | 0.5308 | 0.6145 | 0.5696 | 0.8813 |

HistGradientBoosting tiene el mejor equilibrio entre Recall, F1 y ROC-AUC, así que es la
familia elegida. Las tres familias superan al baseline en F1 y ROC-AUC.

## Modelo final

Búsqueda de hiperparámetros sobre HistGradientBoosting, también en validation:

```text
learning_rate = 0.10
max_iter = 200
max_leaf_nodes = 31
min_samples_leaf = 20
classification_threshold = 0.35
```

| Partición | Modelo | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Validation | Persistencia (baseline) | 0.4468 | 0.4469 | 0.4468 | 0.7288 |
| Validation | HistGradientBoosting | 0.5165 | 0.6941 | 0.5923 | 0.8893 |
| Test | Persistencia (baseline) | 0.4391 | 0.4390 | 0.4391 | 0.7218 |
| Test | HistGradientBoosting | 0.4898 | 0.6871 | 0.5719 | 0.8786 |

En test, HistGradientBoosting mejora al baseline en Recall (+0.248), F1 (+0.133), ROC-AUC
(+0.157) y Precision (+0.051). Detecta 69 % de las horas de riesgo, contra 44 % de la regla
de persistencia.

## MLflow y Model Registry

Los runs se registran en el experimento único `mibici-equipo02`, con `question=P2`,
`model_family`, `stage_of_work`, `git_commit`, `data_version`, fechas de corte, percentil de
riesgo y semilla.

| Run | `stage_of_work` |
|---|---|
| `p2-baseline-persistencia-p80` | `baseline` |
| `P2-HistGradientBoosting-candidate` (y uno por familia) | `candidate` |

El modelo elegido se registra como `mibici-equipo02-p2-riesgo` con el alias `champion`:

```bash
python -m mi_bici.modeling.register_p2 --run-id <RUN_ID>
```

`register_p2` solo acepta una candidata de P2 terminada con `git_commit` y `data_version`. La
versión registrada guarda como tags el run de origen, el commit, la versión de los datos, las
fechas de corte, el umbral de riesgo y los hiperparámetros del modelo. Los Run ID y el número
de versión cambian al reproducir el experimento en otro servidor de MLflow.

## Reproducción

Desde la raíz del repositorio, con `features_p2.parquet` generado y MLflow disponible:

```bash
python -m mi_bici.modeling.baseline_p2
python -m mi_bici.modeling.train --question p2 --model default
```

`--model default` entrena la familia de `P2_MODEL_FAMILY` en `mi_bici/config.py`
(HistGradientBoosting). Para otra familia se usa `logistic_regression`,
`hist_gradient_boosting` o `random_forest`.

## Pruebas

Las pruebas de P2 usan datos sintéticos y corren en CI sin los datos de MiBici. Su detalle está
en la sección «Pruebas y chequeos» del README:

| Archivo | Qué prueba |
|---|---|
| `tests/test_baseline_p2.py` | Regla de persistencia y sus métricas. |
| `tests/test_features_p2.py` | Construcción de la tabla estación-hora y sus variables. |
| `tests/test_train_p2.py` | Split temporal, etiqueta de riesgo, `evaluate_model` y ablaciones. |
| `tests/test_reproducibility_p2.py` | Reproducibilidad de las tres familias, con caso de control. |
| `tests/test_register_p2.py` | Linaje exigido al registrar el modelo. |

## Limitaciones

- El modelo aproxima el riesgo con el desbalance entre salidas y llegadas; no conoce el
  inventario real de bicicletas de cada estación.
- El umbral P80 depende de la distribución de train y define un riesgo relativo, no un umbral
  físico de la estación.
- Test cubre un solo mes (junio de 2026), así que no mide la estabilidad en otras temporadas.
