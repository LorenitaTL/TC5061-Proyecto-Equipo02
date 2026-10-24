# P1 — Resultados de la Entrega 2

## Objetivo y protocolo

Predecir los viajes iniciados por estación-hora en MiBici.

Se mantuvieron las particiones cronológicas:
- Entrenamiento: enero de 2025 a marzo de 2026.
- Validación: abril y mayo de 2026.
- Prueba: junio de 2026.

Las tres candidatas utilizan una muestra reproducible de 1,000,000 de
observaciones del entrenamiento, con RANDOM_STATE=42 y las mismas variables.
La selección se realizó con validación; posteriormente se evaluó únicamente
la candidata seleccionada sobre prueba, sin reentrenarla.

## Configuraciones y resultados de validación

| Modelo | max_iter | learning_rate | max_leaf_nodes | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Baseline estacional | — | — | — | 0.963986 | 1.944943 |
| A: HistGradientBoosting | 100 | 0.10 | 31 | 0.836037 | 1.436694 |
| B: HistGradientBoosting | 200 | 0.05 | 31 | **0.835743** | **1.435471** |
| C: HistGradientBoosting | 100 | 0.10 | 15 | 0.838372 | 1.442953 |

Las candidatas mantienen loss="squared_error" y l2_regularization=1.0.

El baseline utiliza lag_168h y, cuando falta, lag_24h, lag_1h y cero,
en ese orden.

## Comprobación en horas pico

Las horas pico se definieron como las cuatro horas del día con mayor
promedio de viajes por estación-hora en entrenamiento. Se evaluaron esas
horas sobre la partición de validación.

| Modelo | MAE en horas pico | RMSE en horas pico |
|---|---:|---:|
| Baseline | 1.619855 | 2.667839 |
| A | **1.357363** | 2.034820 |
| B | 1.357416 | **2.032926** |
| C | 1.359896 | 2.043338 |

Esta comprobación describe el desempeño en horas de mayor actividad.
No constituye una evaluación de variabilidad entre semillas o muestras.

## Selección

Se seleccionó B por su menor MAE global de validación, utilizando RMSE
y el desempeño en horas pico como criterios complementarios.

B mejora aproximadamente 13.3 % en MAE y 26.2 % en RMSE frente al baseline
en validación. En horas pico mejora aproximadamente 16.2 % en MAE.

La diferencia de MAE en horas pico entre A y B es de apenas 0.000053
viajes por observación. No se interpreta como una diferencia práctica
relevante ni como evidencia de significancia estadística.

El costo de B es mayor: ejecutó 189 iteraciones frente a 97 de A,
debido a la detención anticipada del estimador. Las duraciones totales
observadas fueron aproximadamente 55 y 53 segundos, respectivamente.
Son mediciones de una ejecución local, que incluyen lectura, evaluación
y registro; no son un benchmark.

## Evaluación final en junio

Se recuperó B desde MLflow y se evaluó sin reentrenamiento sobre
289,902 observaciones.

| Modelo | MAE de prueba | RMSE de prueba |
|---|---:|---:|
| Baseline | 1.0175 | 2.0772 |
| B seleccionado | **0.8747** | **1.5501** |

Valores redondeados a cuatro decimales. B mejora aproximadamente 14.0 %
en MAE y 25.4 % en RMSE frente al baseline.

El conjunto de prueba no se utilizó para elegir entre A, B y C.

## Corridas de referencia

Experimento: `mibici-equipo02`.

| Ejecución | Run ID |
|---|---|
| Baseline | `3f88171998734250935d7e5c8827dfbe` |
| A | `c2e1a7eb4e7a40a0b3aad28bf92d3152` |
| B | `6338e74554154891ae7b4a59ba479f57` |
| C | `7a4af041d67749b4bd54ee8b3c548f9d` |
| Evaluación final de B | `3806d997bd124bfea4f03910cc03f487` |

Estos identificadores pertenecen al servidor local donde se ejecutaron
las pruebas. Las nuevas ejecuciones generarán identificadores propios.

## Artefactos y trazabilidad

Se comprobó en las tres candidatas:
- Manifiesto en `data/data_manifest.json`.
- Configuración efectiva en `config/run_config.json`.
- Gráficas de validación real vs. predicho y residuales en `plots/`.
- Conjunto de entrenamiento registrado mediante `mlflow.log_input`.
- Modelo `modelo_p1` con firma y carga exitosa.

El entrenamiento también registra un ejemplo de entrada del modelo.
El conjunto registrado corresponde a la muestra realmente utilizada,
con los tipos empleados para entrenar.

La versión de datos se deriva del manifiesto. El digest de MLflow del
conjunto de entrenamiento complementa esa referencia para identificar
la tabla concreta utilizada.

## Reproducción del entrenamiento

Precondiciones: dependencias instaladas, dataset analítico de P1
generado con la ingeniería de variables del proyecto y servidor de
MLflow disponible.

En PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Baseline
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p1 -P model=seasonal

# Candidata A
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p1 -P model=histgb -P sample_size=1000000 -P max_iter=100 -P learning_rate=0.1 -P max_leaf_nodes=31 -P l2_regularization=1.0

# Candidata B
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p1 -P model=histgb -P sample_size=1000000 -P max_iter=200 -P learning_rate=0.05 -P max_leaf_nodes=31 -P l2_regularization=1.0

# Candidata C
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p1 -P model=histgb -P sample_size=1000000 -P max_iter=100 -P learning_rate=0.1 -P max_leaf_nodes=15 -P l2_regularization=1.0

# Evaluación final del ganador de estas ejecuciones
python -m mi_bici.modeling.evaluate_p1 --model-uri "runs:/6338e74554154891ae7b4a59ba479f57/modelo_p1"
```

En una reproducción, sustituir el Run ID de evaluación por el del
ganador seleccionado en las nuevas corridas.

## Model Registry

Registro local completado:
- Nombre: `mibici-equipo02-p1-demanda`.
- Versión: `1`.
- Alias: `champion`.
- Origen: candidata B.
- URI: `models:/mibici-equipo02-p1-demanda@champion`.

Se documentaron en la versión el experimento, corrida de origen, commit,
versión de datos y corrida de evaluación final. Se verificó la carga por
alias y la igualdad de predicciones con el modelo original sobre diez
observaciones.

El alias expresa el modelo seleccionado para esta entrega. No implica
que exista un servicio desplegado.

Para aplicar la misma política de predicción utilizada en la evaluación,
usar `predict_demand` de `evaluate_p1.py`, que convierte las variables a
float32 y recorta las predicciones negativas a cero.

La consolidación de corridas y Registry en el servidor único del equipo
queda a cargo de Claudio. Los IDs y números de versión pueden cambiar.

## Limitaciones y pendientes de integración

- Los viajes representan demanda observada, no demanda insatisfecha.
- No se dispone del historial completo de disponibilidad de estaciones.
- Las horas completadas con cero pueden incluir periodos de inactividad.
- La evaluación es retrospectiva, con rezagos históricos disponibles antes
  de cada predicción; no equivale a un pronóstico multihorizonte sin
  actualización de los rezagos.
- Integrar la configuración centralizada que incorporará Claudio.
- Completar revisión del PR y reproducción final desde un clon limpio.
