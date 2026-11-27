# Model Card — `mibici-equipo02-p2-riesgo`

Clasificador de riesgo operacional por estación-hora para MiBici (pregunta P2 del proyecto
TC5061, Equipo 02). Esta ficha describe el modelo que sirve la API (`POST /predict`). Los
valores técnicos vienen de la [ficha del champion](p2_ficha_champion.md), que se comprueba
contra el Model Registry con `make champion-card-p2`.

## 1. Identificación

| Campo | Valor |
| :--- | :--- |
| Modelo registrado | `mibici-equipo02-p2-riesgo` |
| Alias en producción | `champion` (MLflow 3 usa aliases en lugar de stages) |
| Versión | `1` del Registry compartido, registrada el 01/11/2026 a las 20:40 hora local según la interfaz de MLflow (`mlflow_p2_champion_linaje.png`) |
| Familia | `HistGradientBoosting` dentro de un pipeline `SimpleImputer` → `StandardScaler` → clasificador |
| Hiperparámetros | `learning_rate=0.10`, `max_iter=200`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `random_state=42` |
| Umbral de clasificación | `0.35` (guardado como `classification_threshold` en el run y en los tags de la versión) |
| Run de origen | `55d8255b3272402ea2532729ec2b1629`, experimento `mibici-equipo02` |
| Linaje | `git_commit`, `data_version` y `features_data_version` como tags de la versión ([ficha](p2_ficha_champion.md)) |
| Fecha de esta ficha | 26/11/2026 |
| Owner del modelo | Eduardo Valencia Guzmán ([@Eduardo-Valencia96](https://github.com/Eduardo-Valencia96)) |
| Revisión | Ariadna Meliza Martínez Anaya ([@MelizaMartinez98](https://github.com/MelizaMartinez98)) |
| Contacto | Issues del repositorio [LorenitaTL/TC5061-Proyecto-Equipo02](https://github.com/LorenitaTL/TC5061-Proyecto-Equipo02/issues) |

## 2. Propósito y usuarios

**Uso previsto.** Apoyar al equipo de rebalanceo de MiBici a decidir a qué estaciones enviar
camiones en la siguiente hora. Para cada estación y hora `t`, el modelo estima la probabilidad
de que el desbalance relativo de la hora `t+1` esté en el 20 % más alto observado en
entrenamiento. La salida es una priorización: una lista de estaciones con mayor probabilidad
de desbalance alto.

- **Horizonte:** una hora. La predicción solo es útil si llega antes de que termine la hora
  actual, por eso se sirve en tiempo real.
- **Usuarios:** operadores de rebalanceo y planeación operativa de MiBici.
- **Decisión que apoya:** el orden de visita de estaciones; la decisión final la toma una
  persona con información adicional (tráfico, camiones disponibles, eventos).

**Fuera de alcance.** No debe usarse para:

- decisiones de seguridad personal de usuarios o trabajadores;
- sanciones, evaluación de desempeño de personas o de zonas;
- afirmar cuántas bicicletas o anclajes habrá físicamente disponibles: el modelo no conoce el
  inventario real de la estación, solo el flujo de viajes (ver §6);
- horizontes distintos de una hora, otras ciudades o estaciones nuevas sin validación adicional.

## 3. Datos

| Aspecto | Detalle |
| :--- | :--- |
| Fuente | Datos abiertos de viajes de MiBici Guadalajara (`references/data_manifest.json`: 18 archivos mensuales, 6,622,260 viajes, con SHA-256 por archivo) |
| Periodo | Enero 2025 a junio 2026 |
| Unidad de análisis | Estación-hora; 480 estaciones |
| Variables de entrada | 30 numéricas (`FEATURE_COLUMNS` en `mi_bici/modeling/train_p2.py`): actividad actual (`departures`, `arrivals`, `net_flow`), desbalance actual y capacidad, ubicación, calendario y codificación cíclica, rezagos de 1, 2 y 24 h y ventanas móviles de 3 y 24 h |
| Etiqueta `Risk` | `Risk = 1` si `|D(t+1) − A(t+1)| / capacidad > 0.066667`, el P80 calculado **solo en train**. La comparación es estricta (`>`) por empates en el P80 |
| Split temporal | Train 2025-01-01 → 2026-03-31 (5,241,600 filas, 19.40 % positivos); validation 2026-04-01 → 2026-05-31 (702,720 filas, 17.65 %); test 2026-06-01 → 2026-06-30 (345,600 filas, 17.31 %) |
| Faltantes | 41,760 celdas en train, solo en rezagos y ventanas móviles; se imputan dentro del pipeline (`SimpleImputer`) |

Test se usó una sola vez, para la evaluación final y el A/B; no participó en la selección de
familia, hiperparámetros ni umbral.

## 4. Modelo y evaluación

**Selección.** Se compararon tres familias con el mismo pipeline contra un baseline de
persistencia (predice riesgo si el desbalance actual ya supera el umbral). En validation se
eligió, para cada familia, el umbral con mejor F1 entre los que alcanzan precisión ≥ 0.50.

| Modelo (validation) | Umbral | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Persistencia (baseline) | — | 0.4468 | 0.4469 | 0.4468 | 0.7288 |
| Logistic Regression | 0.70 | 0.5181 | 0.4395 | 0.4756 | 0.8334 |
| **HistGradientBoosting** | **0.35** | 0.5158 | 0.6910 | 0.5907 | 0.8882 |
| Random Forest | 0.55 | 0.5308 | 0.6145 | 0.5696 | 0.8813 |

**Métricas del champion por partición** (umbral 0.35):

| Partición | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0.8296 | 0.5463 | 0.7170 | 0.6201 | 0.8933 | — |
| Validation | 0.8313 | 0.5165 | 0.6941 | 0.5923 | 0.8893 | — |
| Test | 0.8219 | 0.4898 | 0.6871 | 0.5719 | 0.8786 | 0.5666 |
| Test, baseline | — | 0.4391 | 0.4390 | 0.4391 | 0.7218 | — |

En test el modelo detecta 69 % de las horas de riesgo contra 44 % de la persistencia. Recall
es la métrica principal: una estación de riesgo no atendida cuesta más que una visita de más.
La caída de validation a test es pequeña y consistente con evaluar un periodo posterior.

**A/B testing.** La versión 2 (challenger, igual salvo `learning_rate=0.05`) se comparó con
el champion sobre junio de 2026 (345,600 filas, 480 estaciones) con McNemar e IC 95 % por
bootstrap de estaciones (2,000 réplicas). La regla de promoción, fijada antes de ver
resultados, exige que B mejore Recall de forma significativa sin empeorar F1. B obtuvo un
ΔRecall negativo (−0.0032, IC 95 % de −0.0061 a −0.00002) y un ΔF1 negativo (−0.0020, IC 95 %
de −0.0032 a −0.0007), con McNemar p = 0.0028, por lo que no cumplió el criterio y **se
conservó A como champion**. Los valores salen sin redondear de
`reports/ab_test_p2/ab_metrics.csv`; en `ab_metrics.md` el límite superior de Recall aparece
como −0.0000 por el redondeo a cuatro decimales.

## 5. Contrato de operación

| Aspecto | Detalle |
| :--- | :--- |
| Entrada | JSON con las 30 variables de `FEATURE_COLUMNS` (`mi_bici/modeling/train_p2.py`), todas obligatorias y numéricas, sin columnas extra (`extra="forbid"` en `mi_bici/api/schemas.py`, ejemplo en `references/api_predict_example.json`). Error 422 si no cumple |
| Salida | `risk_probability` (0–1), `risk_class` (1 si `probability ≥ 0.35`), `risk_label` (`alto`/`bajo`), `threshold`, `model_name`, `model_version` |
| Carga | Al arrancar, la API resuelve `models:/mibici-equipo02-p2-riesgo@champion` y lee el umbral del run de origen. `MODEL_VERSION` fija una versión concreta |
| Falla de carga | La API arranca sin modelo: `GET /health` responde `model_loaded: false` y `POST /predict` responde **503**, nunca predice con un modelo equivocado |
| Rendimiento medido | 1 réplica, CPU: cero fallas en las tres cargas de Locust; ~28 RPS con p95 de 620 ms (u50) y saturación cerca de 30 RPS con p95 de 2.5 s (u100) (`reports/load_test/resumen.md`) |
| Costo de referencia | ~30 USD/mes por réplica 24/7; u50 es el mejor compromiso latencia/costo (`reports/finops/e3_finops.md`) |

**Modelo semilla.** `docker compose` sin override registra un modelo **sintético** con el
mismo pipeline (tag `origen=seed-sintetico`) para que CI y las pruebas de carga funcionen sin
datos. Sus latencias son representativas; sus predicciones no. No se presenta como este modelo.

## 6. Limitaciones y sesgos

- **Proxy de riesgo.** `Risk` mide desbalance de flujo de viajes relativo a la capacidad, no
  estaciones vacías o llenas. Sin inventario real no se puede afirmar disponibilidad física.
- **Umbral relativo.** El P80 define "alto" respecto a la historia de train; si el sistema
  cambia (más estaciones, otra demanda), el 20 % superior deja de significar lo mismo.
- **Precisión moderada.** Con el umbral 0.35, cerca de la mitad de las alertas son falsos
  positivos (precision 0.49). Es una elección a favor de Recall que el operador debe conocer.
- **Estacionalidad.** El entrenamiento cubre 15 meses; eventos atípicos (vacaciones, lluvias,
  eventos masivos) están poco representados y no hay variables de clima ni eventos.
- **Calidad y cobertura geográfica.** Depende de que los datos abiertos lleguen completos.
  Las estaciones con poca actividad tienen menos señal; el desempeño no se ha medido por
  estación ni por zona, así que puede ser desigual entre zonas de la ciudad.
- **Drift.** En junio, 0 de 30 predictores superan el umbral de drift de 0.1; la tasa de
  riesgo baja de 19.4 % a 17.3 %. Solo se mide drift marginal de datos; no hay monitoreo de
  drift de concepto (cambio en la relación variables → riesgo) sin etiquetas.

## 7. Monitoreo

- **Frecuencia:** mensual, con `make drift-p2` (Evidently, referencia = train, mes actual =
  mes nuevo, 50,000 filas por conjunto). Resultados en `reports/monitoring_p2/`.
- **Qué vigilar:** las ventanas de flujo de 24 h (`net_flow_rolling_24h`,
  `departures_rolling_24h`, `arrivals_rolling_24h`). Las de calendario (`day_of_week`,
  `dow_cos`) se excluyen de la alerta porque cambian con la mezcla de días de cada mes.
- **Alerta:** investigar si una variable de flujo supera 0.1 o si más del 30 % de los
  predictores tiene drift.
- **Antes de reentrenar:** medir Recall y F1 del champion con etiquetas reales del mes. El
  drift por sí solo no justifica reentrenar.
- **En operación:** `/health` y cada respuesta de `/predict` incluyen `model_version`, para
  confirmar qué versión responde.

## 8. Gobernanza

**Regla de promoción.** Una versión nueva se registra con
`python -m mi_bici.modeling.register_p2 --run-id <RUN_ID> --alias challenger`; el script exige
un run candidato de P2 terminado, con `git_commit` y `data_version`, y comprueba que el modelo
registrado prediga igual que el original. El challenger pasa a champion solo si en el A/B
(`python -m mi_bici.monitoring.ab_test_p2 --model-a champion --model-b challenger`) su Recall
es significativamente mayor (IC 95 % de ΔRecall > 0) y su F1 no es significativamente menor.

**Promoción y reversión.** Se mueve el alias, no se registra de nuevo:

```bash
make promote-p2 VERSION=<versión> DRY_RUN=1   # valida sin mover el alias
make promote-p2 VERSION=<versión>
```

El comando rechaza versiones que no sean de P2, sin `git_commit` o `data_version`, o que sean
el modelo semilla sintético, y comprueba que el alias quedó en la versión pedida. Después se
reinicia el contenedor de la API, que carga el modelo una sola vez al arrancar. Para revertir
se corre el mismo comando con la versión anterior (el comando la imprime) y se reinicia. Después de cualquier
cambio se corre `make champion-card-p2` y se actualiza esta ficha.

**Evidencia de revisión.** Cada cambio llega por PR con revisión de otro integrante y CI en
verde. La decisión A/B queda como run de MLflow (`stage_of_work = ab_test`) y en
`reports/ab_test_p2/ab_summary.json`.

**Historial de versiones.**

| Versión | Fecha | Cambio | Alias | Decisión |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 01/11/2026 | HistGradientBoosting, `learning_rate=0.10`, umbral 0.35 | `champion` | Modelo en producción |
| 2 | 15/11/2026 | Igual que la 1 con `learning_rate=0.05` | `challenger` | Rechazada en el A/B: Recall significativamente menor |

**Aprobación de cambios.** El owner del modelo (Eduardo) propone la promoción con la evidencia
del A/B; otro integrante del equipo la revisa en el PR antes de mover el alias.
