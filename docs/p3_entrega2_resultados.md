# P3 — Resultados de la Entrega 2

## Objetivo y protocolo

Identificar grupos de estaciones de MiBici que comparten patrones de uso y ubicarlos en la
ciudad.

Cada estación se describe con su perfil de uso, calculado en `mi_bici/features_p3.py`:
- `balance_am` y `balance_pm`: balance de flujo en las horas pico de días laborales
  (6-9 h y 17-20 h).
- `balance_dia`: balance de todo el día laboral.
- `share_fin_semana`: proporción de salidas en sábado o domingo, recortada en el percentil
  99 de entrenamiento (0.3614).
- `log_viajes_dia`: volumen; se usa para describir los grupos, no para formarlos.

Se mantuvo la partición de la Entrega 1:
- Entrenamiento: enero a diciembre de 2025, 368 estaciones con al menos 1,000 salidas.
- Prueba: enero a junio de 2026, 374 estaciones con al menos 500 salidas (umbral
  proporcional a la duración del periodo).

Todos los modelos son un pipeline `StandardScaler` + `KMeans` (`n_init=10`,
`RANDOM_STATE=42`) ajustado solo con entrenamiento. Para que las métricas sean comparables,
la silhouette se calcula siempre en el espacio de perfiles de uso (`balance_am`,
`balance_pm`, `share_fin_semana`), aunque el modelo se haya ajustado con otras variables.

Métricas:
- `train_silhouette` / `test_silhouette`: separación de los perfiles de uso.
- `train_stability_ari`: ARI promedio entre el ajuste con la semilla del proyecto y cuatro
  semillas adicionales. Vale 1 si los grupos no dependen de la inicialización.
- `train_min_cluster_size`: estaciones del grupo más pequeño.
- `test_ari_vs_refit`: ARI entre la asignación de 2026 con el modelo de 2025 y un modelo
  reajustado con 2026. Mide si los grupos se sostienen en el tiempo.

La selección se hizo con las métricas de entrenamiento y estabilidad; las de prueba se
reportan para confirmar que los grupos se mantienen.

## Configuraciones y resultados

| Modelo | k | Variables | `train_silhouette` | `train_stability_ari` | Grupo mínimo | `test_silhouette` | `test_ari_vs_refit` |
|---|---:|---|---:|---:|---:|---:|---:|
| Baseline: ubicación | 3 | latitud, longitud | 0.108 | 1.000 | 70 | 0.097 | 1.000\* |
| A: perfil de uso | 3 | `balance_am`, `balance_pm`, `share_fin_semana` | **0.357** | **1.000** | 53 | **0.314** | **0.945** |
| B: perfil de uso | 4 | `balance_am`, `balance_pm`, `share_fin_semana` | 0.273 | 1.000 | 56 | 0.243 | 0.626 |
| C: perfil de uso | 5 | `balance_am`, `balance_pm`, `share_fin_semana` | 0.262 | 0.886 | 24 | 0.230 | 0.493 |
| D: perfil + balance del día | 4 | A + `balance_dia` | 0.283 | 0.974 | 46 | 0.253 | 0.833 |

\* Es trivial: las coordenadas de las estaciones no cambian entre periodos.

## Selección

Se seleccionó A (`kmeans_perfil`, k = 3).

- **Frente al baseline:** la silhouette en el espacio de uso sube de 0.108 a 0.357 en
  entrenamiento y de 0.097 a 0.314 en prueba. La ubicación sola no separa estaciones con
  usos distintos.
- **Frente a más grupos (B y C):** con k = 4 y k = 5 la silhouette baja y los grupos dejan
  de sostenerse en 2026 (ARI de 0.626 y 0.493). Con k = 5 aparece un grupo de 24
  estaciones y la asignación depende de la semilla (ARI 0.886).
- **Frente a agregar `balance_dia` (D):** no mejora la separación y es menos estable que A
  en 2026 (ARI 0.833 frente a 0.945). El balance del día ya queda reflejado en los grupos de
  A (ver el cluster 2).
- **Interpretabilidad:** los tres grupos de A corresponden a perfiles reconocibles para la
  operación (residencial, destino laboral y recreativo).
- **Costo:** todas las configuraciones se ajustan en segundos; el costo no distingue entre
  ellas.

La silhouette de 0.36 es moderada: los perfiles forman un continuo, por lo que los grupos son
una segmentación útil y estable, no grupos naturales bien separados.

### Perfiles de los grupos del modelo seleccionado (entrenamiento)

| Cluster | Estaciones | `balance_am` | `balance_pm` | `balance_dia` | `share_fin_semana` | Salidas/día (mediana) | Perfil |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 178 | -0.280 | 0.186 | 0.026 | 0.206 | 22.9 | Residencial: pierde bicicletas en la mañana y las recupera en la tarde |
| 1 | 137 | 0.205 | -0.205 | -0.056 | 0.177 | 31.2 | Destino laboral: gana bicicletas en la mañana y las pierde en la tarde |
| 2 | 53 | -0.051 | -0.167 | -0.109 | 0.279 | 15.4 | Recreativo: más uso en fin de semana y pérdida sistemática en el día |

El cluster 2 es el único que pierde bicicletas de forma sistemática, por lo que necesita
reabastecimiento aunque tenga menos volumen. Los clusters 0 y 1 se compensan a lo largo del
día, de modo que la redistribución entre ellos se concentra en las horas pico.

## Corridas de referencia

Experimento: `mibici-equipo02`. Commit: `1af96e4`.

| Ejecución | Run ID |
|---|---|
| Baseline | `14e68773e5e94b0383a3c4b7b219fffd` |
| A (seleccionado) | `1e9365a28f714e069b313587b9ff6230` |
| B | `08407868eb2043d092a0c75a356ed353` |
| C | `e2f225566bf940e68b0cccf12e99f757` |
| D | `4e09875e858e4f428d6e146680ea141d` |

Estos identificadores pertenecen al servidor local donde se ejecutaron las pruebas. Las
nuevas ejecuciones generarán identificadores propios.

## Artefactos y trazabilidad

Cada corrida registra:
- Tags `team`, `question`, `git_commit`, `data_version`, `model_family=kmeans` y
  `stage_of_work` (`baseline` o `candidate`).
- Modelo `modelo_p3` (pipeline `StandardScaler` + `KMeans`) con firma e input example.
- Manifiesto en `data/data_manifest.json` y conjunto de entrenamiento mediante
  `mlflow.log_input` (`p3_station_profiles_train`).
- Configuración efectiva en `config/run_config.json`.
- Resumen por cluster en `data/cluster_summary.csv` y asignación de cada estación en
  `data/station_clusters.csv` (entrenamiento) y `data/station_clusters_test.csv` (prueba).
- Mapa y perfil de los grupos en `plots/clusters.png` y `plots/clusters_test.png`.

Los perfiles se agregan por estación; no contienen datos de usuarios.

## Reproducción

Precondiciones: dependencias instaladas, ingesta ejecutada (`mlflow run . -e ingest`) y
servidor de MLflow disponible.

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Baseline
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local \
  -P question=p3 -P model=kmeans_ubicacion

# Candidatas A, B, C y D
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local \
  -P question=p3 -P model=kmeans_perfil -P n_clusters=3
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local \
  -P question=p3 -P model=kmeans_perfil -P n_clusters=4
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local \
  -P question=p3 -P model=kmeans_perfil -P n_clusters=5
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local \
  -P question=p3 -P model=kmeans_perfil -P n_clusters=4 \
  -P features=balance_am,balance_pm,balance_dia,share_fin_semana
```

En PowerShell, usar `$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"`. En macOS, el
puerto 5000 puede estar ocupado por AirPlay Receiver; en ese caso, levantar el servidor con
`--port 5001` y ajustar la URI.

## Model Registry

Registro local completado:
- Nombre: `mibici-equipo02-p3-grupos`.
- Versión: `1`.
- Alias: `champion`.
- Origen: candidata A.
- URI: `models:/mibici-equipo02-p3-grupos@champion`.

La versión documenta en su descripción y en sus tags el experimento, la corrida de origen,
el commit, la versión de datos, las variables, el número de grupos y los periodos de
entrenamiento y prueba. Se verificó que el modelo original, la versión registrada y el alias
asignan los mismos grupos a las 374 estaciones de prueba guardadas en el run.

```bash
python -m mi_bici.modeling.register_p3 --run-id 1e9365a28f714e069b313587b9ff6230
```

El script solo acepta corridas candidatas de P3 terminadas y reutiliza la versión existente
si el mismo run ya está registrado. En otro servidor, sustituir el Run ID por el de la nueva
corrida A.

El alias expresa el modelo seleccionado para esta entrega. No implica que exista un servicio
desplegado.

## Limitaciones

- Los perfiles forman un continuo; la silhouette moderada indica una segmentación útil, no
  grupos naturales bien separados.
- Las estaciones abiertas después de 2025 no tienen perfil de entrenamiento; se asignan con
  el modelo, pero no participaron en su ajuste.
- Los balances reflejan viajes realizados; no incluyen la redistribución que hace el
  operador ni la disponibilidad de bicicletas.
- La consolidación de corridas y Registry en el servidor único del equipo queda a cargo de
  Claudio. Los IDs y números de versión pueden cambiar.
