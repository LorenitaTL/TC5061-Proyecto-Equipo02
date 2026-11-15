# P2: drift y calidad de datos con Evidently (entrenamiento vs. junio de 2026)

## Alcance y reproducción

Desde la raíz del repositorio, con el entorno principal (`requirements.txt`, que ahora
incluye `evidently==0.7.23`):

    make drift-p2

Equivale a:

    python -m mi_bici.monitoring.prepare_p2 --input data/processed/features_p2.parquet
    python -m mi_bici.monitoring.drift_p2

Requisito: `data/processed/features_p2.parquet` (lo genera `python -m mi_bici.features_p2`
a partir de la ingesta). Las salidas quedan en `reports/monitoring_p2/`:

| Archivo | Contenido |
|---|---|
| `p2_drift_calidad.html` | Reporte de Evidently: `DataDriftPreset` (drift por variable) y `DataSummaryPreset` (calidad) |
| `p2_drift_por_variable.csv` | Método, umbral, puntaje y decisión por variable |
| `p2_drift_resumen.json` | Versión de Evidently, ventanas, semilla, muestra, calidad de ambos conjuntos |
| `p2_drift_por_variable.png` | Gráfica 1: puntaje de drift por variable con el umbral |
| `p2_distribuciones_drift.png` | Gráfica 2: tasa de `Risk` y distribuciones de las 3 variables con mayor puntaje |

Conjuntos comparados:

- **Referencia**: periodo de entrenamiento del modelo A (champion), enero de 2025 a
  marzo de 2026 (`P2_TRAIN_END = 2026-04-01`, exclusivo): 5,241,600 filas estación-hora.
- **Actual**: conjunto común de evaluación A/B de junio de 2026 (`prepare_p2`, PR #111):
  345,600 filas.

Ambos usan las 30 variables de la ablación `completo` y la etiqueta `Risk` con el mismo
umbral de entrenamiento (`target_imbalance_ratio > 0.0667`, percentil 80 de train).
Evidently recibe una muestra aleatoria de 50,000 filas por conjunto (semilla 42); las
cifras de calidad del JSON se calculan sobre los conjuntos completos.

Métodos fijados en el código (no los elige Evidently según el tamaño de la muestra):

- Numéricas: distancia de Wasserstein normalizada por la desviación estándar de la referencia.
- Categóricas (`hour_of_day`, `day_of_week`, `is_weekend`, `Risk`): distancia de Jensen-Shannon.
- Hay drift cuando el puntaje es **≥ 0.1**, el valor por defecto de Evidently. Es un umbral
  exploratorio, no calibrado con el impacto en el desempeño del modelo.

## Resultados

**Ningún predictor (0/30) ni la etiqueta `Risk` supera el umbral.** El conjunto de
junio no presenta drift según este criterio.

| Variable | Método | Puntaje |
|---|---|---:|
| net_flow_rolling_24h | Wasserstein normalizada | 0.086 |
| dow_cos | Wasserstein normalizada | 0.079 |
| departures_rolling_24h | Wasserstein normalizada | 0.069 |
| net_flow_rolling_3h | Wasserstein normalizada | 0.065 |
| arrivals_rolling_24h | Wasserstein normalizada | 0.060 |
| imbalance_ratio | Wasserstein normalizada | 0.047 |
| Risk | Jensen-Shannon | 0.017 |
| capacity, latitude, longitude | Wasserstein normalizada | < 0.010 |

La tabla completa está en `p2_drift_por_variable.csv`.

Contexto para leer los puntajes (cálculo complementario con las mismas muestras de 50,000):

- **Ruido de muestreo**: dos muestras independientes del propio entrenamiento dan
  puntajes de ~0.01. Los valores de junio (0.05-0.09) son un cambio real, aunque pequeño.
- **Abril y mayo** (validación) muestran cambios del mismo orden; en mayo,
  `departures_rolling_24h` llegó a 0.12. Junio no es atípico frente a los meses previos.
- **`net_flow_rolling_24h` crece mes a mes**: 0.048 (abril), 0.062 (mayo), 0.079 (junio,
  0.086 en el reporte). Es la variable más cercana al umbral.

Robustez: con las semillas 7 y 2026 ningún predictor supera el umbral; el máximo queda
entre 0.075 y 0.086 y las variables en los primeros lugares son las mismas.

## Calidad de los datos

| Indicador | Referencia | Junio |
|---|---:|---:|
| Filas | 5,241,600 | 345,600 |
| Estaciones | 480 | 480 |
| Claves estación-hora duplicadas | 0 | 0 |
| Celdas faltantes en las 30 variables | 41,760 | 0 |
| Proporción `Risk = 1` | 19.4 % | 17.3 % |

- Los faltantes de la referencia están solo en los 15 rezagos y ventanas móviles
  (`*_lag_*`, `*_rolling_*`) y corresponden a la falta de historia al inicio de cada
  estación. Junio no tiene faltantes. El pipeline del modelo los imputa (`SimpleImputer`).
- Las mismas 480 estaciones aparecen en ambos conjuntos; `capacity`, `latitude` y
  `longitude` prácticamente no cambian (puntaje < 0.01). No hay altas, bajas ni
  reubicaciones que afecten la cobertura del modelo.
- La sección de calidad del HTML (`DataSummaryPreset`) muestra, por variable, conteos,
  faltantes, valores únicos y estadísticas descriptivas de ambos conjuntos.

## Interpretación operativa

1. **El modelo A (champion) puede seguir operando en junio** y la comparación A/B con el
   candidato B es válida: ambos se evalúan sobre datos que siguen pareciéndose a los de
   entrenamiento. Este reporte no justifica por sí solo reentrenar.
2. **`dow_cos` y `day_of_week` reflejan el calendario, no un cambio de comportamiento.**
   Junio de 2026 empieza en lunes y tiene 30 días: hay 5 lunes y 5 martes y solo 4 de
   cada otro día, por lo que la mezcla de días difiere de la de 15 meses completos. Esta
   variación se repetirá cada mes y no debe disparar alertas.
3. **La tasa de riesgo baja de 19.4 % a 17.3 %** (2.1 puntos; Jensen-Shannon 0.017). Con
   un umbral de decisión fijo, menos positivos tienden a reducir la precisión observada.
   Al comparar A y B en junio conviene reportar también métricas por clase (precision,
   recall) y no solo F1 o accuracy, y comparar contra la prevalencia de 17.3 %.
4. **Lo que hay que vigilar son las ventanas de 24 h de flujo** (`net_flow_rolling_24h`,
   `departures_rolling_24h`, `arrivals_rolling_24h`). En junio el flujo neto de 24 h es
   más equilibrado: las horas con |flujo neto| > 0.25 bajan de 22.9 % a 17.8 % y la
   desviación estándar de 0.377 a 0.305. Además, las horas sin salidas en 24 h bajan de
   23.7 % a 19.2 %. Hay actividad más repartida y menos desequilibrio sostenido, lo que
   concuerda con la menor tasa de `Risk`. Si la tendencia continúa en julio y estas
   variables superan 0.1, hay que revisar el desempeño del champion con etiquetas reales
   antes de decidir reentrenar.
5. **Regla propuesta para el monitoreo mensual**: correr `make drift-p2` con cada mes
   nuevo; investigar si alguna variable de flujo supera 0.1 o si más del 30 % de los
   predictores muestra drift; reentrenar solo si además baja el desempeño medido con
   etiquetas. Las variables de calendario se excluyen de la alerta por lo explicado en (2).

## Limitaciones

- Se comparan distribuciones marginales globales; no se mide la estabilidad por estación
  ni las relaciones entre variables (drift de concepto).
- El drift de datos no mide desempeño: el efecto real sobre el modelo se evalúa con las
  métricas A/B sobre `Risk` en junio.
- El umbral de 0.1 es el valor por defecto de Evidently, no está calibrado con el costo
  operativo de una alerta.
- Con muestras pequeñas el ruido de muestreo puede acercarse al umbral (en las pruebas
  sintéticas con 500 filas llega a ~0.1); por eso se usan 50,000 filas por conjunto.

## Pruebas

`tests/test_drift_p2.py` (6 pruebas, sin depender de `data/`):

- La referencia respeta el corte de entrenamiento y etiqueta `Risk` con el umbral de train.
- El muestreo es reproducible y acotado.
- Los tipos nullable (`Float64`, `Int64`) se convierten a tipos que Evidently acepta.
- Si faltan columnas del modelo, el script falla con un mensaje claro.
- Con datos idénticos no hay drift; con un desplazamiento deliberado, sí. En ambos casos
  se generan los cinco archivos de salida.
