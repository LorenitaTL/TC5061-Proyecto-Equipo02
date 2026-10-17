# P1. Predicción de demanda horaria de MiBici Guadalajara

## 1. Objetivo

Desarrollar un modelo de regresión para predecir la cantidad de viajes que se iniciarán en cada estación de MiBici durante una hora determinada, utilizando información histórica de viajes y variables temporales.

La unidad de análisis es la combinación **estación-hora** y la variable objetivo es `trip_count`, definida como el número de viajes iniciados en esa estación durante la hora correspondiente.

## 2. Datos y preparación

Se utilizaron 18 archivos mensuales de viajes, correspondientes al periodo de enero de 2025 a junio de 2026, previamente normalizados mediante el pipeline de ingeniería de datos del proyecto.

El conjunto contiene 6,622,260 viajes, agregados en 4,862,675 observaciones estación-hora.

Para construir el dataset analítico:

1. Se contabilizaron los viajes por estación de origen y hora de inicio.
2. Se completaron con cero las horas sin viajes registrados, desde la primera actividad observada de cada estación hasta el final del periodo.
3. Se generaron variables calendario y rezagos históricos.
4. Se almacenó el resultado en `data/processed/p1_demanda_horaria.parquet`.

### Variables predictoras

| Variable | Descripción |
|---|---|
| `hour_of_day` | Hora del día, de 0 a 23 |
| `day_of_week` | Día de la semana, de 0 a 6 |
| `is_weekend` | Indicador de fin de semana |
| `lag_1h` | Viajes iniciados una hora antes |
| `lag_24h` | Viajes iniciados 24 horas antes |
| `lag_168h` | Viajes iniciados 168 horas antes |

Los rezagos se calcularon por estación, respetando el orden cronológico. Los valores faltantes al inicio de cada serie se conservaron para distinguir la ausencia de historia de la demanda observada igual a cero.

## 3. Diseño experimental

Se establecieron particiones cronológicas:

| Conjunto | Periodo |
|---|---|
| Entrenamiento | Enero 2025 – marzo 2026 |
| Validación | Abril – mayo 2026 |
| Prueba | Junio 2026 |

Se empleó una evaluación retrospectiva de predicción horaria. Para cada hora objetivo, se permite utilizar información de horas anteriores, incluyendo observaciones reales inmediatamente anteriores pertenecientes al periodo evaluado.

### Baseline estacional

Se utilizó como referencia el número de viajes registrados en la misma hora de la semana anterior (`lag_168h`). Ante la ausencia de ese valor se utiliza `lag_24h`, después `lag_1h` y finalmente cero.

### Modelo candidato

Se entrenó un `HistGradientBoostingRegressor` de scikit-learn, con las siguientes configuraciones:

- Función de pérdida: `squared_error`.
- Iteraciones máximas: 100.
- Número máximo de hojas: 31.
- Tasa de aprendizaje: 0.1.
- Regularización L2: 1.0.
- Semilla aleatoria: 42.

Se compararon dos tamaños de muestra reproducibles del periodo de entrenamiento: 300,000 y 1,000,000 de observaciones. La segunda configuración se seleccionó por sus menores errores en validación.

No se incluyó `station_id` como predictor, por lo que el modelo aprende patrones compartidos entre estaciones a partir de las variables temporales y los rezagos.

## 4. Resultados

| Métrica | Baseline validación | Modelo V1 validación | Modelo V2 validación |
|---|---:|---:|---:|
| MAE | 0.9640 | 0.8391 | **0.8360** |
| RMSE | 1.9449 | 1.4637 | **1.4367** |

En la evaluación final de junio de 2026 se obtuvieron los siguientes resultados:

| Métrica | Baseline | Modelo V2 |
|---|---:|---:|
| MAE | 1.0175 | **0.8748** |
| RMSE | 2.0772 | **1.5508** |

El modelo V2 redujo el MAE de prueba aproximadamente un 14.0 % y el RMSE un 25.3 % respecto al baseline.

Los resultados respaldan una mejora predictiva sobre el conjunto analítico construido. No demuestran que se haya estimado demanda insatisfecha.

## 5. Trazabilidad con MLflow

Se utilizó MLflow Tracking con SQLite como backend local.

Experimento: `MiBici_P1_Demanda_Horaria`.

Se registraron parámetros, métricas, metadatos y el modelo entrenado. La evaluación final recuperó el modelo desde MLflow sin reentrenarlo.

Los identificadores de las corridas y modelos registrados localmente sirven como evidencia de ejecución, pero no sustituyen el procedimiento reproducible, puesto que otros integrantes pueden obtener identificadores distintos.

## 6. Ejecución

Instalar las dependencias del repositorio y disponer previamente de los archivos mensuales Parquet procesados.

Desde la raíz del proyecto:

```bash
python -m mi_bici.features_p1
python -m mi_bici.modeling.baseline_p1
python -m mi_bici.modeling.train_p1
```

Para evaluar un modelo previamente registrado:

```bash
python -m mi_bici.modeling.evaluate_p1 --model-uri models:/IDENTIFICADOR_DEL_MODELO
```

Para visualizar los experimentos:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Para ejecutar las comprobaciones:

```bash
python -m ruff check .
python -m pytest -q
python -m pip check
```

## 7. Limitaciones

- Los datos permiten predecir viajes iniciados, no demanda potencial insatisfecha.
- No se cuenta con información histórica completa sobre apertura, cierre o disponibilidad efectiva de las estaciones.
- Las horas completadas con cero pueden incluir periodos durante los cuales una estación no estaba operativa.
- Algunas estaciones aparecen por primera vez cerca del final del periodo, por lo que cuentan con poca historia.
- El modelo no incorpora características espaciales, meteorológicas ni identificadores explícitos de estación.
- Las métricas corresponden a una evaluación retrospectiva con actualización de información horaria; no equivalen a realizar desde una fecha inicial todas las predicciones del mes siguiente.

## 8. Pruebas automatizadas

Se incorporaron pruebas para verificar:

- Generación de horas sin viajes y conservación de conteos.
- Separación correcta entre estaciones.
- Cálculo de rezagos históricos.
- Construcción de variables calendario.
- Límites y exclusividad de las particiones cronológicas.

Las pruebas se ejecutan mediante `pytest`, mientras que Ruff permite verificar las reglas de calidad de código del repositorio.
