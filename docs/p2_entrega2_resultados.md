# P2 — Resultados de la Entrega 2

> **Nota:** este documento conserva los resultados de la Entrega 2. Después, siguiendo los
> comentarios del tutor, P2 cambió: el baseline oficial es la regla de persistencia (la
> regresión logística es una candidata más), la etiqueta usa `>` en lugar de `>=` (19.40 % de
> positivos en train), la partición usa las mismas fechas de corte que P1, se comparan tres
> familias de modelos y el modelo elegido es HistGradientBoosting. Los resultados vigentes
> están en la sección
> [P2 — Predicción de riesgo operacional](../README.md#p2--predicción-de-riesgo-operacional)
> del README.

## Objetivo y protocolo

Predecir el riesgo operacional por estación-hora en MiBici, identificando aquellas observaciones en las que el desbalance entre llegadas y salidas representa una situación de riesgo para la operación de la estación.

El análisis utiliza el conjunto `features_p2.parquet`, con **6,289,920 observaciones**.

Se mantuvieron particiones cronológicas para evitar fuga de información temporal:

- **Entrenamiento:** 4,402,560 observaciones, del 2025-01-01 al 2026-01-18 03:00:00.
- **Validación:** 943,680 observaciones, del 2026-01-18 04:00:00 al 2026-04-10 01:00:00.
- **Prueba:** 943,680 observaciones, del 2026-04-10 02:00:00 al 2026-06-30 23:00:00.

La selección de modelos se realizó utilizando exclusivamente la partición de validación. La partición de prueba se reservó para la evaluación final de la candidata seleccionada.

La métrica primaria de selección fue **Recall**, debido a que el objetivo operativo es identificar correctamente las estaciones con riesgo.

---

## Definición del riesgo

El riesgo se define a partir del `target_imbalance_ratio`.

El umbral de riesgo se estableció en el percentil 80 (**P80**) calculado exclusivamente sobre el conjunto de entrenamiento:

```text
risk_threshold = 0.066667
```

Este umbral se calculó sobre TRAIN y posteriormente se aplicó sin modificación a validation y test.

La etiqueta se construye como:

```text
Risk = 1 si target_imbalance_ratio >= risk_threshold
Risk = 0 en otro caso
```

Esto permite mantener una definición consistente del riesgo entre las tres particiones y evita utilizar información de validación o prueba para definir la etiqueta.

---

## Variables utilizadas

El modelo utiliza variables relacionadas con el flujo y el contexto temporal y espacial de cada estación.

Entre las principales variables se encuentran:

- `departures`
- `arrivals`
- `net_flow`
- `absolute_imbalance`
- `imbalance_ratio`
- `capacity`
- `latitude`
- `longitude`
- `hour_of_day`
- `day_of_week`
- `is_weekend`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- variables lag
- variables rolling

Se evaluaron además diferentes ablations para estudiar la importancia de las variables relacionadas con el flujo y el desbalance actual.

---

## Configuraciones y resultados de validación

Se utilizó **Regresión Logística** como familia de modelo.

| Modelo / experimento | Configuración | Validation Recall | Validation Precision | Validation F1 | Validation ROC-AUC |
|---|---|---:|---:|---:|---:|
| Baseline | Completo + `class_weight=balanced` | 0.7391 | 0.5597 | 0.6370 | 0.8410 |
| Candidata | Completo + sin balanceo | 0.4454 | 0.6742 | 0.5364 | 0.8382 |
| **Candidata seleccionada** | Sin desbalance actual + `class_weight=balanced` | **0.7417** | 0.5470 | 0.6296 | 0.8373 |
| Candidata | Sin flujo actual + `class_weight=balanced` | 0.7395 | 0.5333 | 0.6197 | 0.8294 |

Las ablations fueron:

- `completo`: utiliza todas las variables.
- `sin_desbalance_actual`: elimina `absolute_imbalance` e `imbalance_ratio`.
- `sin_flujo_actual`: elimina las variables relacionadas con el flujo actual.

---

## Selección

Se seleccionó la candidata **sin desbalance actual + `class_weight=balanced`**, debido a que obtuvo el mayor Recall de validación:

```text
Recall validation = 0.7417
```

La selección priorizó Recall porque, desde el punto de vista operacional, es más importante detectar correctamente las situaciones de riesgo que maximizar únicamente la precisión de las alertas.

La comparación con el baseline muestra:

- Recall de validación: **0.7391 → 0.7417**
- F1 de validación: 0.6370 → 0.6296
- ROC-AUC de validación: 0.8410 → 0.8373

Por lo tanto, la selección representa un intercambio entre una ligera mejora en Recall y una reducción moderada en Precision, F1 y ROC-AUC.

La diferencia de Recall es pequeña, por lo que no se interpreta como evidencia de una mejora estadísticamente significativa. La selección responde al criterio operativo definido para P2.

---

## Evaluación final sobre TEST

Una vez seleccionada la candidata utilizando validation, se realizó la evaluación final sobre TEST.

| Métrica | TEST |
|---|---:|
| Recall | **0.7313** |
| Precision | 0.5397 |
| F1 | 0.6210 |
| ROC-AUC | 0.8330 |

El desempeño de Recall se mantiene cercano al observado en validation:

```text
Validation Recall = 0.7417
Test Recall       = 0.7313
```

La diferencia entre ambas particiones es aproximadamente 0.0104 puntos. El conjunto de prueba no fue utilizado para seleccionar la configuración del modelo.

---

## Corridas de referencia

Experimento: `mibici-equipo02`.

### Baseline

`b482323fed5a4954804dd4ba263f5f67`

Configuración:

```text
model_family = LogisticRegression
stage_of_work = baseline
question = P2
ablation = completo
```

### Búsqueda de hiperparámetros

Run padre: `20d9441e98e8428bb26a0d38d18fbf06`

| Configuración | Run ID |
|---|---|
| Completo + balanced | `34f167ba947c4a63b0d307c7a2dc085e` |
| Completo + sin balanceo | `76cccaba13874af9a3235d56a090d6df` |
| Sin desbalance actual + balanced | `892101e0ed51438a871b668a849c1c47` |
| Sin flujo actual + balanced | `0f4882c57b99467ea6e122dd35235db2` |

La candidata seleccionada corresponde a `892101e0ed51438a871b668a849c1c47`.

La evaluación final registrada en MLflow corresponde al Run ID `d1dfcb2ca7514e4c88b0e3388152b2de`.

---

## Artefactos y trazabilidad

El entrenamiento P2 registra en MLflow:

- Modelo entrenado.
- Firma del modelo mediante `infer_signature`.
- `input_example`.
- Manifiesto y versión de datos.
- Registro explícito del conjunto de entrenamiento mediante `mlflow.log_input(...)`.
- Configuración efectiva en `config/`.
- Métricas de entrenamiento, validación y prueba.
- Umbral de riesgo P80.
- Matriz de confusión.
- Curva Precision-Recall.
- Artefactos de métricas.
- Artefactos relacionados con el modelo y la evaluación.

Las gráficas generadas incluyen:

```text
plots/confusion_matrix_validation.png
plots/precision_recall_curve_validation.png
plots/confusion_matrix_test.png
plots/precision_recall_curve_test.png
```

La llamada a `mlflow.log_input(...)` utiliza el conjunto de entrenamiento de P2 y registra información de trazabilidad del dataset, incluyendo su nombre, fuente y perfil.

---

## Reproducción del entrenamiento

Precondiciones:

- Dependencias instaladas.
- Dataset `features_p2.parquet` disponible.
- Entorno virtual activo.
- Servidor MLflow disponible.

En PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

### Entrenamiento baseline

```powershell
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local `
  -P question=p2 -P model=default
```

### Búsqueda de hiperparámetros

```powershell
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local `
  -P question=p2 -P model=search
```

La implementación utiliza el contrato:

```python
run(model, params)
```

---

## Model Registry

El modelo seleccionado fue registrado en MLflow Model Registry con:

- **Nombre:** `mibici-equipo02-p2-riesgo`
- **Source Run:** `892101e0ed51438a871b668a849c1c47`
- **Versión:** 3
- **Alias:** `champion`
- **URI:** `models:/mibici-equipo02-p2-riesgo@champion`

Se verificó la carga del modelo mediante el alias `champion`.

Para registrar una nueva candidata:

```powershell
python -m mi_bici.modeling.register_p2 --run-id <RUN_ID>
```

### Versiones para el A/B testing

Para la Actividad 10 se registró una segunda versión del mismo modelo (candidata B) que solo
cambia `learning_rate` de HistGradientBoosting. La corrida final se ejecutó en un servidor de
MLflow con las dos versiones registradas:

| Versión | Alias | `learning_rate` | Source Run |
|---|---|---|---|
| A: 1 | `champion` | 0.10 | `55d8255b3272402ea2532729ec2b1629` |
| B: 2 | `challenger` | 0.05 | `ac91bbd5a7804b7d99687684a3245e6d` |

- A y B usan los mismos datos (`features_data_version` `337f806a…`), el mismo split
  temporal, el mismo umbral de riesgo P80 (0.066667) y el mismo umbral de clasificación
  (0.35). La única diferencia es `learning_rate`.
- Al registrar B, `register_p2` verificó que `models:/mibici-equipo02-p2-riesgo@challenger`
  carga y produce las mismas predicciones que el modelo del run de origen.
- `champion` siguió apuntando a la versión 1 después de registrar B, así que la API no
  cambió.
- Los comandos para reproducir el entrenamiento y el registro de B están en el README
  («Candidata B para el A/B testing»). Los números de versión y los run ID cambian si se
  reproduce en otro servidor de MLflow; el entrenamiento usa semilla fija, así que las
  métricas no.

### Resultado del A/B testing

Comparación con `mi_bici.monitoring.ab_test_p2` sobre junio de 2026 (345,600 horas-estación
de 480 estaciones; 17.31 % con Risk = 1). Junio es el periodo de test de P2, posterior al
entrenamiento, y ninguna versión lo usó para entrenar ni para elegir hiperparámetros.

| Métrica | A (v1) | B (v2) | Δ B − A | IC 95 % de Δ |
|---|---:|---:|---:|---|
| Recall | 0.6871 | 0.6839 | −0.0032 | [−0.0061, −0.00002] |
| Precision | 0.4898 | 0.4885 | −0.0013 | [−0.0025, 0.00003] |
| F1 | 0.5719 | 0.5699 | −0.0020 | [−0.0032, −0.0007] |
| Accuracy | 0.8219 | 0.8213 | −0.0006 | — |
| ROC-AUC | 0.8786 | 0.8775 | −0.0011 | — |
| PR-AUC | 0.5666 | 0.5669 | +0.0004 | — |

- IC 95 % con bootstrap por estación (2,000 réplicas, semilla 42).
- McNemar: en 2,599 filas solo acierta A y en 2,387 solo acierta B (p = 0.0028).
- Las métricas calculadas coinciden exactamente con las de test que registró cada run en
  MLflow (diferencia máxima 0), lo que confirma que el comparador reproduce la evaluación del
  entrenamiento.

**Decisión: se conserva A (v1) como `champion`.** Según la regla fijada antes de la
comparación, B solo reemplaza a A si mejora Recall de forma significativa; B no lo mejora.
Su Recall es menor, aunque el límite superior del intervalo queda apenas por debajo de 0, y
su F1 es significativamente menor. Las diferencias son estadísticamente detectables por el
tamaño de la muestra, pero pequeñas en la práctica (menos de medio punto porcentual): bajar
`learning_rate` de 0.10 a 0.05 no aporta mejora y A sigue vigente.

Los archivos de la comparación están en `reports/ab_test_p2/` y la corrida está registrada en
MLflow como `P2-ab-test-v1-vs-v2` (`stage_of_work = ab_test`).

---

## Política de clasificación

El modelo genera probabilidades mediante:

```python
model.predict_proba(X)[:, 1]
```

Actualmente la implementación utiliza la clasificación estándar de `scikit-learn`, sin una selección explícita de un umbral alternativo de probabilidad basado exclusivamente en validation.

Por otra parte, el **umbral de riesgo P80 = 0.066667** utilizado para construir la variable objetivo sí se calcula exclusivamente sobre TRAIN.

Estas dos nociones de umbral se mantienen separadas:

- **Umbral de riesgo:** define qué observaciones son consideradas riesgo.
- **Umbral de clasificación:** convierte la probabilidad estimada por el modelo en una predicción binaria.

No se implementó un umbral alternativo de clasificación basado en validation, ya que la implementación actual utiliza la clasificación estándar de `scikit-learn` y el P80 del riesgo es un umbral distinto, utilizado para construir la variable objetivo.

---

## Limitaciones y pendientes de integración

- El modelo identifica riesgo operacional con base en el desbalance observado, pero no representa necesariamente toda la disponibilidad física futura de las estaciones.
- El umbral P80 depende de la distribución del `target_imbalance_ratio` en el conjunto de entrenamiento.
- La selección de la candidata mejora Recall sólo ligeramente respecto al baseline.
- La mejora de Recall no se interpreta como evidencia de significancia estadística.
- La selección explícita de un umbral de clasificación basado únicamente en validation no está implementada.
- Los Run ID y números de versión pueden cambiar al reproducir el experimento en otro servidor MLflow.
- La consolidación final del servidor MLflow y Registry depende de la infraestructura compartida del equipo.

---

## Checklist de evidencia

| Requisito | Estado |
|---|---|
| Split temporal train/validation/test | ✅ |
| Definición de riesgo con P80 sobre TRAIN | ✅ |
| Modelo baseline | ✅ |
| Búsqueda de candidatas | ✅ |
| Selección utilizando validation | ✅ |
| Evaluación final sobre TEST | ✅ |
| Recall como métrica primaria | ✅ |
| `mlflow.log_model` | ✅ |
| Firma del modelo | ✅ |
| `input_example` | ✅ |
| Manifiesto / versión de datos | ✅ |
| `mlflow.log_input()` desde P2 | ✅ |
| `config/` | ✅ |
| Matriz de confusión | ✅ |
| Curva Precision-Recall | ✅ |
| Model Registry | ✅ |
| Alias `champion` | ✅ |
| Contrato `run(model, params)` | ✅ |

---

## Conclusiones

El experimento P2 implementa un modelo de clasificación de riesgo operacional para MiBici utilizando Regresión Logística y una partición temporal que evita utilizar información futura durante el entrenamiento.

La selección se realizó sobre validation utilizando Recall como métrica primaria. La candidata seleccionada, que elimina las variables de desbalance actual y utiliza balanceo de clases, obtuvo un Recall de **0.7417 en validation** y **0.7313 en TEST**.

El modelo se encuentra integrado con MLflow, incluyendo trazabilidad de configuración, métricas, firma, input example, registro del conjunto de entrenamiento mediante `mlflow.log_input(...)`, artefactos de evaluación y Model Registry. La candidata seleccionada se encuentra registrada como `mibici-equipo02-p2-riesgo` con alias `champion`.

El modelo no utiliza un umbral alternativo de clasificación seleccionado exclusivamente sobre validation. El umbral P80 de **0.066667** corresponde a la definición del riesgo y se calcula únicamente sobre TRAIN.
