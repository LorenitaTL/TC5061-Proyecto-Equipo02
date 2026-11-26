# Ficha técnica del champion de P2

Responsable: [@Eduardo-Valencia96](https://github.com/Eduardo-Valencia96). Revisión: [@LorenitaTL](https://github.com/LorenitaTL).

> Esta es la única fuente de verdad sobre qué modelo de P2 es el champion. README, Model Card y
> presentación deben citar estos valores. Para comprobarlos contra el Model Registry:
>
> ```bash
> make champion-card-p2   # escribe reports/p2_champion_registry.md y falla si algo no coincide
> ```
>
> Los valores de abajo se tomaron de las evidencias del Registry compartido
> (`reports/figures/evidencia/mlflow_p2_champion_linaje.png` y
> `mlflow_p2_ab_versiones.png`) y del resumen del A/B (`reports/ab_test_p2/ab_summary.json`).
> Los hashes completos de `git_commit` y `data_version` se verificaron fuera del Registry: el
> commit existe en el historial de Git y `data_version` se recalcula desde `references/data_manifest.json`.

| Campo | Valor confirmado | Fuente |
| :--- | :--- | :--- |
| Registered model | `mibici-equipo02-p2-riesgo` | MLflow Registry |
| Alias operativo | `champion` | MLflow Registry |
| Versión | `1`, registrada el 01/11/2026 20:40 (hora local de la interfaz de MLflow) | Model version (`mlflow_p2_champion_linaje.png`, columna *Registered at*); `ab_summary.json` → `A.version` |
| Familia | `HistGradientBoosting` (pipeline `SimpleImputer` + `StandardScaler` + clasificador) | Tag `model_family` |
| Umbral de clasificación | `0.35` | Tag `classification_threshold`; `P2_CLASSIFICATION_THRESHOLD` en `mi_bici/config.py` |
| Hiperparámetros | `learning_rate=0.1`, `max_iter=200`, `max_leaf_nodes=31`, `min_samples_leaf=20` | Tags de la versión; `P2_HGB_*` en `mi_bici/config.py` |
| Run de origen | `55d8255b3272402ea2532729ec2b1629` (`P2-HistGradientBoosting-candidate`) | Tag `source_run_id` |
| Commit | `cb67db696e9316e41e4dc20929f4511b5e99504c` | Tag `git_commit` (prefijo en la captura); el commit existe en Git: «fix: atender los comentarios del tutor pendientes en P2», 01/11/2026 20:28, 12 minutos antes del registro |
| Versión de datos | `198acbc4c315b252808c926b4a7e165853610f487ff630d3087c4fd36401ee8d` | Tag `data_version` (prefijo en la captura); se reproduce con `get_data_version()` de `mi_bici/modeling/tracking.py` sobre `references/data_manifest.json` |
| Versión de features | `337f806ade6654cd1c21f42e0339…` (prefijo visible en la captura; valor completo en el tag de la versión 1 del Registry compartido) | Tag `features_data_version`: SHA-256 de `data/processed/features_p2.parquet`. El archivo no está en Git y al regenerarse puede cambiar de bytes (por ejemplo, por la versión de pyarrow), así que una copia local puede tener otro hash aunque los datos de origen (`data_version`) sean los mismos |
| Ventanas | Train 2025-01-01 → 2026-03-31; validation 2026-04-01 → 2026-05-31; test 2026-06-01 → 2026-06-30 | Tags `train_end=2026-04-01`, `validation_end=2026-06-01` |
| Etiqueta Risk | `target_imbalance_ratio > 0.066667` (P80 de train, comparación estricta) | Tags `risk_percentile=0.8`, `risk_threshold` |
| Origen | Entrenado con datos reales de MiBici (sin tag `origen=seed-sintetico`) | Tags de la versión |
| Estado de producción | La versión a la que apunta `champion`. MLflow 3 usa aliases en lugar de stages (`Staging`/`Production`) | README y Registry |
| Challenger evaluado | Versión `2`, run `ac91bbd5a7804b7d99687684a3245e6d`, igual salvo `learning_rate=0.05`. Perdió el A/B; `champion` no se movió | `ab_summary.json` |

## Métricas de test del run de origen (junio 2026)

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|
| 0.8219 | 0.4898 | 0.6871 | 0.5719 | 0.8786 | 0.5666 |

Las cinco primeras son las `test_*` registradas en el run de origen; el A/B las recalculó y
coinciden exactamente (`max_gap_vs_logged_test = 0.0`). PR-AUC la calcula el A/B.

## Qué no es el champion

| Registro | Qué es | Cómo reconocerlo | Cómo presentarlo |
| :--- | :--- | :--- | :--- |
| Versión 1 del Registry compartido | **El champion real** | `model_family=HistGradientBoosting`, run `55d8255b…`, sin tag `origen` | Es el modelo que se reporta y se defiende. |
| Modelo semilla de Compose (`mi_bici/operations/seed_model.py`) | Mismo pipeline e hiperparámetros, entrenado con 5,000 filas **sintéticas** | Tag `origen=seed-sintetico`; `make champion-card-p2` falla con «No es el modelo semilla sintético» | Solo para operar el sistema y medir carga. Nunca como el mejor modelo. |
| Registros locales de cada integrante (por ejemplo la versión 5 de la evidencia E4) | Copias en el `mlflow.db` personal; el número de versión depende de cuántas veces se registró en esa máquina | El número de versión no es 1 y el run no es `55d8255b…` | No citar su número de versión en la documentación. |

## Antes de grabar la demo

1. Apuntar `MLFLOW_TRACKING_URI` al Registry que se usará (o levantar
   `docker-compose.local.yml` con el `mlflow.db` real).
2. Correr `make champion-card-p2`. Debe terminar con todas las verificaciones en OK.
3. Revisar que `GET /health` y `POST /predict` respondan el mismo `model_version` que la ficha.
   Si responde la semilla, la narración debe decirlo explícitamente.
