# mi-bici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Propósito

El proyecto busca anticipar la operación de MiBici, el sistema de bicicleta pública de
Guadalajara, a partir de los tiempos, orígenes y destinos de sus viajes: predecir la demanda
de cada estación, identificar riesgos operativos, como quedarse sin bicicletas o sin anclajes
disponibles, y encontrar grupos de estaciones con patrones de uso similares.

## Preguntas del proyecto

| ID | Pregunta de negocio | Tipo de tarea | Quién la usaría |
| :--- | :--- | :--- | :--- |
| **P1** | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Operador del sistema, para planear la flota. |
| **P2** | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Clasificación | Equipo de rebalanceo de bicicletas. |
| **P3** | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | Clustering | Planeación urbana y expansión del sistema. |

## Integrantes y roles

| Integrante | Matrícula | GitHub | Rol |
| :--- | :--- | :--- | :--- |
| Ariadna Meliza Martínez Anaya | A01797975 | [@MelizaMartinez98](https://github.com/MelizaMartinez98) | Ingesta reproducible, calidad y versionado de datos; P1 (regresión) |
| Claudio Salvador Rivas Anguiano | A01840294 | [@clsrivas](https://github.com/clsrivas) | Configuración centralizada y dependencias; baseline y ablaciones de P2; documentación |
| Lorena Trujillo Landeros | A01797858 | [@LorenitaTL](https://github.com/LorenitaTL) | P3 (clustering); integración de MLflow; flujo de trabajo con Git, pre-commit y CI |
| Eduardo Valencia Guzmán | A01631738 | [@Eduardo-Valencia96](https://github.com/Eduardo-Valencia96) | P2 (clasificación); ML Canvas y arquitectura |

## Setup

The project requires Python 3.12. We use [uv](https://docs.astral.sh/uv/) to create the
virtual environment (install it with `brew install uv` on macOS, or see the
[installation guide](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
make create_environment
source .venv/bin/activate
make requirements
```

This installs the dependencies and the [pre-commit](https://pre-commit.com/) git hooks
(`make setup-hooks` installs only the hooks).

Activate the environment with `source .venv/bin/activate` every time you open a new
terminal to work on the project.

If you use Anaconda, run `conda deactivate` before activating `.venv`. With the `base`
environment also active, `mlflow` and `curl` can resolve to Anaconda's copies instead of the
pinned versions. Check with `which mlflow python`: both must point to `.venv/bin/`.

### Windows (without `make` or `uv`)

If you don't have `make` or `uv`, create the environment with the standard library's
`venv` and `pip`. Make sure Python 3.12 is installed (`py -3.12 --version`):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
pre-commit install
```

Activate the environment with `.venv\Scripts\activate` every time you open a new
terminal. Without `make`, run the equivalent commands directly:

| `make` target   | Equivalent command                                                                  |
| --------------- | ----------------------------------------------------------------------------------- |
| `make test`     | `python -m pytest tests`                                                            |
| `make coverage` | `python -m pytest tests --cov=mi_bici --cov-report=term-missing`                    |
| `make security` | `bandit -r mi_bici; pip-audit -r requirements.txt; pip-audit -r requirements-validation.txt` |
| `make create_validation_environment` | `uv venv .venv-validation --python 3.12; uv pip install --python .venv-validation -r requirements-validation.txt` |
| `make validate-data` | `.venv-validation\Scripts\python -m mi_bici.data_validation`                   |
| `make test-validation` | `.venv-validation\Scripts\python -m pytest tests/test_data_validation.py`    |
| `make lint`     | `ruff format --check; ruff check`                                                   |
| `make format`   | `ruff check --fix; ruff format`                                                     |
| `make data`     | `python -m mi_bici.dataset main`                                                    |
| `make clean`    | `Get-ChildItem -Recurse -Directory -Filter __pycache__ \| Remove-Item -Recurse -Force` |

### Pre-commit

On every `git commit`, the hooks in `.pre-commit-config.yaml` run on the staged files:

- `ruff` lints (with `--fix`) and formats the code, using the config in `pyproject.toml`.
- Generic checks: trailing whitespace, end of file, YAML/TOML syntax, merge conflict
  markers, private keys, and files larger than 1 MB.
- `pytest` runs the test suite when the commit includes Python files. It uses the
  Python of the active environment, so activate `.venv` before committing.

If a hook modifies files, the commit is aborted. Review the changes, `git add` them, and
commit again. To run the hooks on the whole repo:

```bash
pre-commit run --all-files
```

## Git workflow

We follow [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow):
`main` always works, and every change goes through a branch and a Pull Request (PR).

`main` is protected: nobody can push to it directly, and every PR needs one approval
before it can be merged.

### Step by step

```bash
# 1. Start from an up-to-date main
git checkout main
git pull

# 2. Create a branch for your change
git checkout -b feat/peak-hour-features

# 3. Work and commit (as many commits as you need)
git add .
git commit -m "feat: add peak hour features"

# 4. Push the branch
git push -u origin feat/peak-hour-features

# 5. Open the PR
gh pr create --base main --fill
```

Then:

6. Ask a teammate to review the PR.
7. If the reviewer asks for changes, commit them on the same branch and `git push` again.
   The PR updates automatically.
8. Once the PR is approved and the `pre-commit` and `tests` checks are green, merge it
   with **Squash and merge** and delete the branch.
9. Go back to step 1 for your next change.

### Branch names

`type/short-description`, using the same types as the commit messages, in lowercase and
with hyphens instead of spaces:

```
feat/data-ingestion
fix/station-encoding
docs/ml-canvas
```

### Commit messages

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
`type: description`. A hook rejects the commit if the message doesn't match.

| Type       | Use it for                                  |
|------------|---------------------------------------------|
| `feat`     | New functionality                           |
| `fix`      | Bug fix                                     |
| `docs`     | Documentation only                          |
| `test`     | Adding or changing tests                    |
| `refactor` | Code change that doesn't change behavior    |
| `chore`    | Anything else (dependencies, config, etc.)  |

Examples:

```
feat: add peak hour features
fix: handle missing station ids
docs: explain how to download the data
```

### Pull Requests

- **Title:** same format as the commit messages (`type: description`). Because PRs are
  merged with *Squash and merge*, the PR title becomes the commit message in `main`.
- **Description:** what changed, why, and how to test it. Link the related issue if
  there is one (`Closes #<issue number>`).
- **Size:** keep PRs small, focused on a single change. They are easier to review.

### Reviews

Every team member opens at least one PR and reviews at least one PR from a teammate.
To review, go to the PR's **Files changed** tab, leave comments on specific lines if
needed, and click **Review changes**:

- **Approve:** the change is ready to merge.
- **Request changes:** something must be fixed first. Explain what and why.
- **Comment:** questions or suggestions that don't block the merge.

All review conversations must be resolved before merging.

## Ingeniería de datos — MiBici

### Objetivo y alcance

Este componente implementa un proceso reproducible para descargar, normalizar y validar los datos abiertos del sistema de bicicletas públicas MiBici de Guadalajara.

Se procesaron **6,622,260 viajes correspondientes a 18 meses**, de enero de 2025 a junio de 2026, mediante procesamiento secuencial para limitar el consumo de memoria.

**Fuentes:**
- [Portal oficial de datos abiertos MiBici](https://mibici.net/es/datos-abiertos/).
- Archivos mensuales de viajes: enero de 2025 a junio de 2026.
- Catálogo de estaciones: `nomenclatura_2026_09.csv`.

### Requisitos e instalación

El proyecto requiere Python 3.12 y las bibliotecas declaradas en requirements.txt. La descarga utiliza curl, cuyo ejecutable se identifica según el sistema operativo (curl.exe en Windows y curl en macOS/Linux). Es necesario que se encuentre instalado y disponible en el PATH.

Desde PowerShell, en la raíz del repositorio:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Ejecución

**1. Descargar y normalizar los datos**

```powershell
python -m mi_bici.dataset ingest --start-month 2025-01 --end-month 2026-06
```

Descubre los archivos mensuales publicados en el portal oficial, descarga y normaliza el catálogo de estaciones y procesa los viajes de manera secuencial.

**2. Generar el manifiesto**

```powershell
python -m mi_bici.dataset manifest
```

Genera `references/data_manifest.json`, con información de procedencia, tamaños de archivos y hashes SHA-256.

**3. Verificar la integridad**

```powershell
python -m mi_bici.dataset verify
```

Compara los hashes actuales de los archivos con los almacenados en el manifiesto.

**4. Generar el reporte de calidad**

```powershell
python -m mi_bici.data_quality
```

Genera `data/processed/data_quality_report.csv`, que combina indicadores de ingesta y correspondencia con el catálogo de estaciones para el periodo completo.

**5. Ejecutar pruebas unitarias**

```powershell
python -m pytest tests/test_dataset_pipeline.py -v
```

### Archivos generados

| Archivo o directorio | Descripción |
|---|---|
| `data/raw/` | CSV mensuales originales |
| `data/external/` | CSV original del catálogo de estaciones |
| `data/processed/viajes_AAAA_MM.parquet` | Viajes mensuales normalizados |
| `data/processed/estaciones.parquet` | Catálogo de estaciones normalizado |
| `data/processed/resumen_ingesta_2025-01_2026-06.csv` | Resumen mensual de ingesta |
| `data/processed/data_quality_report.csv` | Reporte consolidado de calidad |
| `references/data_manifest.json` | Procedencia, tamaños y hashes SHA-256 |
| `references/data_quality_report.csv` | Copia versionable de los resultados de calidad |

Los datos originales y Parquet se mantienen fuera del control de versiones de Git. Los archivos de `references/` contienen metadatos y resultados agregados.

### Normalización y controles de calidad

El pipeline conserva los identificadores de viaje y estación, las fechas de inicio y fin y la duración calculada. Los atributos personales de los usuarios no se transfieren a los archivos normalizados.

Se generan indicadores de fechas inválidas, duraciones negativas, duraciones cero e identificadores de viaje duplicados.

**Principales resultados:**

- 6,622,260 viajes procesados, sin pérdida de registros durante la normalización.
- Sin fechas inválidas, duraciones negativas ni identificadores de viaje duplicados detectados.
- 1,863 viajes con duración cero en febrero de 2026, concentrados principalmente en la segunda mitad del mes.
- Dos identificadores de estación sin correspondencia en el catálogo, 405 y 407, observados en junio de 2026.
- 976 referencias de origen y 1,078 de destino sin correspondencia en el catálogo durante junio de 2026.

Las anomalías se conservan y documentan; no se eliminan registros automáticamente.

### Limitaciones

El catálogo utilizado corresponde a septiembre de 2026, posterior al periodo de estudio. Por tanto, sus capacidades no deben interpretarse automáticamente como capacidades históricas de las estaciones. Esta limitación es especialmente relevante para la estimación del riesgo de falta de bicicletas o espacios disponibles.

Las duraciones cero requieren una decisión de tratamiento específica para cada pregunta de modelado.

Los hashes SHA-256 permiten verificar la integridad de archivos, pero no garantizan por sí mismos la exactitud del contenido ni la disponibilidad futura de las fuentes.

La comprobación de duplicados entre diferentes meses se realizó como validación transversal complementaria.

### Reproducibilidad

La ingesta procesa los archivos mensualmente para reducir la presión sobre la memoria. Durante la ejecución del periodo completo no fue necesario reducir los 18 meses originalmente acordados.

Las versiones directas de las principales dependencias se declaran en `requirements.txt`. Los hashes identifican las versiones exactas de los archivos originales y transformados utilizadas en esta ejecución. Regenerar Parquet con versiones diferentes de las bibliotecas podría modificar sus hashes.

## P1. Predicción de demanda horaria

Se implementó un modelo de regresión para predecir la cantidad de viajes
iniciados por estación y hora, utilizando variables temporales y rezagos
históricos.

El componente incluye la construcción del dataset analítico, un baseline
estacional, entrenamiento y evaluación cronológica, seguimiento de
experimentos con MLflow y pruebas automatizadas.

El modelo seleccionado, HistGradientBoostingRegressor, redujo el MAE de
prueba aproximadamente un 14 % y el RMSE un 25 % respecto al baseline.

**Documentación técnica, resultados e instrucciones de reproducción:**
[Predicción de demanda horaria (P1)](docs/p1_prediccion_demanda.md).

## P2 — Predicción de riesgo operacional

### Objetivo

La pregunta P2 busca predecir si una estación de MiBici presentará un desbalance operacional elevado durante la siguiente hora.

Para cada estación y hora `t`, el modelo utiliza información disponible hasta ese momento para predecir si el desbalance relativo durante la hora `t+1` pertenecerá al 20 % superior de la distribución observada en el conjunto de entrenamiento.

El desbalance relativo de la siguiente hora se define como:

$$
\text{ImbalanceRatio}_{t+1}
=
\frac{|D_{t+1}-A_{t+1}|}{C}
$$

donde:

- $D_{t+1}$ = número de salidas durante la siguiente hora;
- $A_{t+1}$ = número de llegadas durante la siguiente hora;
- $C$ = capacidad de la estación.

El umbral de clasificación se obtiene como el percentil 80 de la distribución del objetivo calculada **únicamente sobre el conjunto de entrenamiento**:

`P80 = 0.066667`

Por lo tanto, una observación pertenece a la clase positiva cuando el desbalance relativo de la siguiente hora supera este umbral. Este valor funciona como un **proxy relativo de riesgo operacional** y no como un umbral físico o de seguridad de una estación.

### Características

El modelo utiliza 30 características, agrupadas en:

- actividad actual: salidas, llegadas y flujo neto;
- desbalance y capacidad de la estación;
- ubicación de la estación;
- variables temporales y variables cíclicas de hora y día;
- rezagos de 1, 2 y 24 horas;
- ventanas móviles históricas de 3 y 24 horas.

Las variables históricas utilizan únicamente información disponible hasta el instante `t`, evitando utilizar información futura para predecir `t+1`.

### Partición temporal

Se utiliza una partición cronológica para evitar fuga de información:

| Conjunto | Periodo | Observaciones |
| --- | --- | ---: |
| Train | 2025-01-01 00:00 → 2026-01-18 03:00 | 4,402,560 |
| Validation | 2026-01-18 04:00 → 2026-04-10 01:00 | 943,680 |
| Test | 2026-04-10 02:00 → 2026-06-30 23:00 | 943,680 |

En total se utilizan **6,289,920 observaciones estación-hora**.

El umbral P80 utilizado para definir la clase positiva se calcula exclusivamente con el conjunto de entrenamiento, evitando utilizar información de validación o prueba durante la definición del problema.

### Pipeline y modelo

El pipeline de clasificación utiliza:

1. imputación de valores faltantes mediante la mediana;
2. estandarización de características;
3. Regresión Logística con `class_weight="balanced"`.

La clasificación se realiza utilizando una probabilidad de decisión de `0.5`.

Las métricas utilizadas son **Recall, Precision, F1 y ROC-AUC**. Recall es especialmente relevante para identificar la mayor cantidad posible de estaciones con riesgo, mientras que Precision y F1 permiten evaluar el costo de las alertas adicionales. ROC-AUC se utiliza como medida complementaria del poder discriminativo del modelo.

### Resultados

La tabla corresponde al baseline con las 30 características. El modelo seleccionado en la
Entrega 2 es la versión sin desbalance actual (ver [Ablaciones](#ablaciones)).

Resultados del baseline sobre los tres conjuntos:

| Conjunto | Recall | Precision | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.7505 | 0.5804 | 0.6546 | 0.8465 |
| Validation | 0.7391 | 0.5597 | 0.6370 | 0.8410 |
| Test | 0.7283 | 0.5524 | 0.6283 | 0.8364 |

El run se registra en el experimento `mibici-equipo02`, compartido por P1, P2 y P3. Se identifica con las etiquetas `team=equipo02`, `question=P2` y `git_commit`.

Los identificadores de los runs dependen de cada ejecución local, por lo que pueden variar entre integrantes.

El entrenamiento puede reproducirse desde la raíz del repositorio mediante:

```bash
python -m mi_bici.modeling.train_p2
```

Los artefactos generados localmente incluyen:

```text
models/p2_risk_model.joblib
models/p2_metrics.csv
models/p2_threshold.txt
```

Estos archivos están excluidos del control de versiones mediante `.gitignore`. El modelo y sus resultados experimentales se registran en MLflow.

### Ablaciones

Se realizaron pruebas de eliminación de grupos de variables para evaluar la contribución de la información de desbalance y actividad actual:

| Modelo | Variables | Recall | Precision | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 30 | 0.7283 | 0.5524 | 0.6283 | 0.8364 |
| Sin desbalance actual | 28 | 0.7313 | 0.5397 | 0.6210 | 0.8330 |
| Sin flujo actual | 25 | 0.7290 | 0.5267 | 0.6116 | 0.8249 |

En la Entrega 2 la selección se hizo solo con validación y con Recall como métrica principal.
La versión **sin desbalance actual** (28 variables, `class_weight="balanced"`) obtuvo el mayor
Recall de validación (0.7417 frente a 0.7391 del baseline), a cambio de una ligera baja en
Precision, F1 y ROC-AUC, por lo que es el modelo seleccionado de P2. Está registrado como
`mibici-equipo02-p2-riesgo` con el alias `champion`. La comparación completa, la búsqueda con
nested runs y la justificación están en
[Resultados de P2 en la Entrega 2](docs/p2_entrega2_resultados.md).

Las variables que quita cada ablación están definidas en `ABLATIONS`, en `mi_bici/modeling/train_p2.py`:

- **Sin desbalance actual:** `absolute_imbalance` e `imbalance_ratio`.
- **Sin flujo actual:** además `departures`, `arrivals` y `net_flow`.

Para reproducir la tabla, cada versión registra su propio run de MLflow (`P2-logistic-regression-baseline`, `P2-logistic-regression-sin-desbalance-actual` y `P2-logistic-regression-sin-flujo-actual`):

```bash
python -m mi_bici.modeling.train_p2 --ablation todas
```

Con `--ablation sin_desbalance_actual` o `--ablation sin_flujo_actual` se entrena solo una. Las ablaciones se registran únicamente en MLflow: los archivos locales de `models/` siempre corresponden al baseline, y el modelo seleccionado se obtiene del Model Registry (`models:/mibici-equipo02-p2-riesgo@champion`).

### Reproducibilidad de P2

Para reproducir el entrenamiento de P2 desde la raíz del repositorio:

```bash
python -m mi_bici.modeling.train_p2
```

El script realiza la carga de las características procesadas, genera la partición temporal, calcula el umbral P80 a partir del conjunto de entrenamiento, entrena el pipeline de clasificación, evalúa el modelo en train, validation y test, y registra el experimento y sus artefactos en MLflow.

## P3 — Grupos de estaciones por perfil de uso

### Formulación

Se agrupan las estaciones según cómo se usan a lo largo del día y de la semana, y se ubican
los grupos en el mapa. La unidad de análisis es la estación. No hay horizonte de
predicción: los grupos describen el periodo de entrenamiento y se valida que se mantengan en
los meses posteriores. Planeación urbana puede usarlos para decidir dónde reforzar la
redistribución y dónde conviene expandir el sistema.

La exploración y la selección del modelo están en
[`notebooks/1.0-lt-perfiles-estaciones.ipynb`](notebooks/1.0-lt-perfiles-estaciones.ipynb).

### Datos y partición

| Conjunto | Periodo | Estaciones |
| --- | --- | ---: |
| Entrenamiento | 2025-01 a 2025-12 | 368 |
| Prueba | 2026-01 a 2026-06 | 374 |

- Los perfiles y el recorte de `share_fin_semana` en su percentil 99 se calculan solo con
  entrenamiento. Los meses de prueba no participan en el ajuste.
- Se descartan los viajes de duración cero (0.06 %), que no representan uso real.
- Se excluyen las estaciones con menos de 1,000 salidas en entrenamiento (500 en los seis
  meses de prueba), que operaron solo parte del periodo y no tienen un perfil completo.

### Variables

Las funciones están en `mi_bici/features_p3.py` (`load_profile_trips`, `flow_balance`,
`build_station_profiles`):

| Variable | Descripción | Uso |
| --- | --- | --- |
| `balance_am` | (llegadas − salidas) / total, de 6 a 9 h en días laborales | Agrupa |
| `balance_pm` | Lo mismo, de 17 a 20 h | Agrupa |
| `share_fin_semana` | Proporción de salidas en sábado o domingo | Agrupa |
| `balance_dia` | Balance de todo el día laboral | Describe los grupos |
| `log_viajes_dia` | log(1 + salidas por día) | Describe los grupos |

Un balance positivo indica que la estación acumula bicicletas (destino) y uno negativo que
se vacía (origen). El volumen no se usa para agrupar porque los grupos separarían estaciones
muy usadas de poco usadas en lugar de perfiles de uso.

### Modelos y métricas

- **Baseline:** K-Means con k = 3 sobre latitud y longitud. Responde si basta la ubicación
  para separar estaciones con usos distintos.
- **Modelo candidato:** K-Means con k = 3 sobre `balance_am`, `balance_pm` y
  `share_fin_semana`, estandarizadas.

Los dos modelos se evalúan en el espacio de perfiles de uso, para que sus métricas sean
comparables:

- `train_silhouette` / `test_silhouette`: qué tan separados están los perfiles de uso.
- `train_stability_ari`: si los grupos cambian con la semilla de K-Means.
- `test_ari_vs_refit`: si los grupos se mantienen en 2026. Compara la asignación del modelo
  de 2025 con un modelo reajustado con los datos de 2026.

### Resultados

| Modelo | `train_silhouette` | `test_silhouette` | `train_stability_ari` | `test_ari_vs_refit` |
| --- | ---: | ---: | ---: | ---: |
| Baseline (ubicación) | 0.108 | 0.097 | 1.000 | 1.000\* |
| Perfil de uso | **0.357** | **0.314** | 1.000 | 0.945 |

\* Es trivial: las coordenadas de las estaciones no cambian entre periodos.

| Cluster | Estaciones | Perfil | Rasgos |
| --- | ---: | --- | --- |
| 0 | 178 | Residencial | Pierde bicicletas en la mañana y las recupera en la tarde. ~23 salidas/día. |
| 1 | 137 | Destino laboral | Gana bicicletas en la mañana y las pierde en la tarde. ~31 salidas/día. |
| 2 | 53 | Recreativo | Más uso en fin de semana (28 %) y pierde bicicletas en el día. ~15 salidas/día. |

El cluster 2 es el único que pierde bicicletas de forma sistemática, por lo que necesita
reabastecimiento aunque tenga menos volumen. Los clusters 0 y 1 se compensan a lo largo del
día, de modo que la redistribución entre ellos se concentra en las horas pico.

### Reproducibilidad de P3

Cada ejecución de `train` registra un run. El modelo `kmeans_ubicacion` es el baseline
(`stage_of_work=baseline`) y `kmeans_perfil`, el valor por defecto, es el candidato. Se
aceptan los hiperparámetros `n_clusters`, `n_init` y `features` (lista separada por comas).

```bash
# Baseline y corridas candidatas
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p3 -P model=kmeans_ubicacion
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p3 -P model=kmeans_perfil -P n_clusters=3
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p3 -P model=kmeans_perfil -P n_clusters=4
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p3 -P model=kmeans_perfil -P n_clusters=5
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p3 -P model=kmeans_perfil -P n_clusters=4 -P features=balance_am,balance_pm,balance_dia,share_fin_semana
```

`python -m mi_bici.modeling.train_p3` registra el baseline y el candidato con los valores por
defecto. Cada run guarda el pipeline (`StandardScaler` + `KMeans`) como `modelo_p3`, con
firma e input example, y como artefactos el manifiesto de datos (`data/data_manifest.json`),
la configuración efectiva (`config/run_config.json`), el resumen por cluster
(`data/cluster_summary.csv`), la asignación de cada estación en entrenamiento y prueba
(`data/station_clusters.csv` y `data/station_clusters_test.csv`) y el mapa de los grupos
junto con su perfil de uso (`plots/clusters.png` y `plots/clusters_test.png`). El conjunto de
entrenamiento se registra con `mlflow.log_input`.

El modelo seleccionado (`kmeans_perfil`, k = 3) se registra como `mibici-equipo02-p3-grupos`
con su linaje y el alias `champion`:

```bash
python -m mi_bici.modeling.register_p3 --run-id <RUN_ID_DE_P3_KMEANS_PERFIL_K3>
```

La comparación de las corridas, la justificación de la selección y el registro se detallan en
[Resultados de P3 en la Entrega 2](docs/p3_entrega2_resultados.md).

### Limitaciones de P3

- Los perfiles forman un continuo, por lo que la silhouette es moderada (0.36). Los grupos son
  una segmentación útil y estable, no grupos naturales bien separados.
- Las estaciones abiertas después de 2025 (401, 402, 405 y 407) no tienen perfil de
  entrenamiento.
- El número de cada cluster depende de la semilla. Para comparar runs se deben usar los
  perfiles, no el número.

## Experimento base en MLflow

Las tres preguntas registran sus runs en un solo experimento, `mibici-equipo02`, en la base
local `mlflow.db`, que se puede cambiar con la variable de entorno `MLFLOW_TRACKING_URI`. Los
artefactos se guardan en `mlartifacts/`. Ambos están excluidos de Git.

Requiere haber ejecutado antes la [ingesta](#ejecución). Desde la raíz del repositorio:

```bash
# P1: dataset, baseline, modelo y evaluación final
python -m mi_bici.features_p1
python -m mi_bici.modeling.baseline_p1
python -m mi_bici.modeling.train_p1
python -m mi_bici.modeling.evaluate_p1 --model-uri "runs:/<RUN_ID_P1>/modelo_p1"

# P2: features, baseline, modelo y ablaciones
python -m mi_bici.features_p2
python -m mi_bici.modeling.baseline_p2
python -m mi_bici.modeling.train_p2
python -m mi_bici.modeling.train_p2 --ablation todas

# P3: perfiles, baseline y modelo por defecto
python -m mi_bici.modeling.train_p3
```

`<RUN_ID_P1>` es el identificador del run `hist_gradient_boosting_v2`. Se usa la forma
`runs:/…/modelo_p1` porque `register_p1` verifica que la evaluación corresponda exactamente a
ese modelo (ver [Registrar el modelo elegido](#5-registrar-el-modelo-elegido-de-cada-pregunta)).

Para abrir la interfaz:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

En macOS, el puerto 5000 suele estar ocupado por AirPlay.

Cada run registra:

| Elemento | Contenido |
| --- | --- |
| Nombre | Descriptivo, por ejemplo `baseline_semanal_v1` o `p3-kmeans-perfil-uso-k3` |
| Etiquetas | `team`, `question` (`P1`, `P2` o `P3`), `git_commit` y `data_version`; desde la Entrega 2, también `model_family` y `stage_of_work` (`baseline`, `search` o `candidate`) |
| Parámetros | `data_version` (huella de `references/data_manifest.json`), rango de fechas, semilla, modelo e hiperparámetros |
| Métricas | Con el conjunto en el nombre: `mae_val`, `mae_test`, `test_recall`, `train_silhouette`, etc. |
| Artefactos | Metadatos y modelo (P1), métricas, umbral y modelo (P2), resumen, asignaciones y mapa de clusters (P3) |

La trazabilidad común está en `mi_bici/modeling/tracking.py`:

- `configure_mlflow()` activa el experimento.
- `log_traceability(question, model_family, stage_of_work)` registra las etiquetas y la
  versión de los datos.
- `log_run_config(config)` guarda la configuración efectiva de la corrida en `config/`.
- `nested_search(question, model_family)` organiza una búsqueda de hiperparámetros como una
  corrida padre con una corrida hija por combinación (ver el ejemplo en su docstring).

## Servidor de tracking y MLflow Project

Desde la Entrega 2, las corridas se lanzan con `mlflow run` contra un servidor local de
tracking. El Model Registry necesita un almacenamiento respaldado por base de datos, por eso
el servidor usa el mismo `mlflow.db`, que conserva los runs de la Entrega 1.

### 1. Levantar el servidor

En una terminal, desde la raíz del repositorio, activa el entorno y levanta el servidor. El
comando del servidor es el mismo en los dos sistemas:

```bash
# macOS / Linux
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

```powershell
# Windows (PowerShell)
.venv\Scripts\activate
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

La interfaz queda en <http://127.0.0.1:5000>. En macOS, si el puerto 5000 está ocupado por
AirPlay, usa otro (por ejemplo, `--port 5001`) y ajústalo también en el paso 2. Deja esta
terminal abierta mientras trabajas.

### 2. Apuntar al servidor

En otra terminal, desde la raíz del repositorio:

```bash
# macOS / Linux
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

```powershell
# Windows (PowerShell)
.venv\Scripts\activate
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

La variable solo dura mientras la terminal esté abierta; hay que definirla de nuevo en cada
terminal nueva. Sin ella, el código escribe directamente en `mlflow.db`, sin pasar por el
servidor.

### 3. Ejecutar con `mlflow run`

El archivo `MLproject` define dos puntos de entrada:

| Punto de entrada | Qué ejecuta | Parámetros |
| --- | --- | --- |
| `ingest` | `python -m mi_bici.dataset main` | Ninguno; el periodo se toma de `mi_bici/config.py` |
| `train` | `python -m mi_bici.modeling.train` | `question` (`p1`, `p2` o `p3`), `model` y los hiperparámetros del modelo |

En la terminal del paso 2:

```bash
# macOS / Linux
# Ingesta
mlflow run . -e ingest --experiment-name mibici-equipo02 --env-manager local

# Entrenamiento: un modelo por ejecución
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local \
  -P question=p3 -P model=kmeans_perfil -P n_clusters=4
```

```powershell
# Windows (PowerShell)
# Ingesta
mlflow run . -e ingest --experiment-name mibici-equipo02 --env-manager local

# Entrenamiento: un modelo por ejecución
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local `
  -P question=p3 -P model=kmeans_perfil -P n_clusters=4
```

En PowerShell, el salto de línea se marca con un acento grave (`` ` ``) en lugar de `\`;
también puedes escribir el comando en una sola línea.

- `--experiment-name mibici-equipo02` es obligatorio: si no coincide con el experimento que
  activa el código, MLflow detiene la corrida.
- Los parámetros que no están declarados en `MLproject` (como `n_clusters`) se pasan al
  modelo de la pregunta. Los valores por defecto viven en `mi_bici/config.py`, no en
  `MLproject`.
- `--env-manager local` usa el entorno ya activado y sirve para desarrollar. Para reproducir
  desde un clon limpio, usa `--env-manager uv`: MLflow crea un entorno nuevo a partir de
  `python_env.yaml`, con las mismas versiones fijas de `requirements.txt`. En Windows, si
  no tienes `uv`, instálalo con `python -m pip install uv`.

Cada pregunta implementa en su `train_pX.py` una función `run(model, params)` que entrena un
solo modelo y abre una sola corrida de MLflow. Los modelos disponibles de cada pregunta se
documentan en su sección.

### 4. Reproducir el modelo elegido de cada pregunta

`ingest` solo genera los viajes y el catálogo de estaciones. P1 y P2 leen además su propio
dataset estación-hora, que hay que construir antes de entrenar. P3 calcula sus perfiles
directamente desde los viajes.

```bash
python -m mi_bici.features_p1   # data/processed/p1_demanda_horaria.parquet
python -m mi_bici.features_p2   # data/processed/features_p2.parquet
```

Después, en la terminal del paso 2 (los comandos funcionan igual en bash/zsh y PowerShell):

```bash
# P1: candidata B (HistGradientBoosting con max_iter=200 y learning_rate=0.05)
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p1 -P model=histgb -P max_iter=200 -P learning_rate=0.05

# P2: búsqueda con nested runs; la hija seleccionada es
# P2-search-C-1.0-weight-balanced-ablation-sin_desbalance_actual
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p2 -P model=search

# P3: K-Means sobre perfiles de uso, k = 3
mlflow run . -e train --experiment-name mibici-equipo02 --env-manager local -P question=p3 -P model=kmeans_perfil -P n_clusters=3
```

- Sin hiperparámetros, `-P question=p1` entrena la candidata A, no la seleccionada.
- En P2, `-P ablation=sin_desbalance_actual` (sin `model=search`) reproduce las mismas métricas
  en una corrida suelta, pero guarda el modelo como `p2_risk_model` y `register_p2` no lo
  acepta: para registrar se usa la hija de la búsqueda.
- En P3, `-P model=kmeans_perfil` es obligatorio aunque sea el valor por defecto: con
  `model=default`, la corrida falla al registrar el parámetro `model`.
- La ingesta regenera `references/data_manifest.json`. No hagas commit de ese cambio; descártalo
  con `git restore references/data_manifest.json`.

### 5. Registrar el modelo elegido de cada pregunta

Cada pregunta tiene un script `register_pX.py` que registra en el Model Registry la versión
elegida, guarda su linaje (experimento, run de origen, `git_commit` y `data_version`) como
etiquetas y descripción de la versión, le asigna el alias `champion` y comprueba que el modelo
cargado por el alias da las mismas predicciones que el original. Los `<RUN_ID>` salen de la
interfaz de MLflow o de la salida de cada `mlflow run`.

```bash
# P1: primero la evaluación final sobre prueba, que register_p1 exige
python -m mi_bici.modeling.evaluate_p1 --model-uri "runs:/<RUN_ID_P1>/modelo_p1"
python -m mi_bici.modeling.register_p1 --run-id <RUN_ID_P1> --evaluation-run-id <RUN_ID_DE_evaluacion_final_p1_v2>

# P2: la hija P2-search-C-1.0-weight-balanced-ablation-sin_desbalance_actual
python -m mi_bici.modeling.register_p2 --run-id <RUN_ID_HIJA_P2>

# P3: el run p3-kmeans-perfil-k3
python -m mi_bici.modeling.register_p3 --run-id <RUN_ID_P3>
```

| Pregunta | Modelo registrado | Se carga con |
| --- | --- | --- |
| P1 | `mibici-equipo02-p1-demanda` | `models:/mibici-equipo02-p1-demanda@champion` |
| P2 | `mibici-equipo02-p2-riesgo` | `models:/mibici-equipo02-p2-riesgo@champion` |
| P3 | `mibici-equipo02-p3-grupos` | `models:/mibici-equipo02-p3-grupos@champion` |

**Esquema de ciclo de vida: aliases, no etapas.** MLflow 3 marca como obsoletas las etapas
(`Staging`, `Production`) y recomienda aliases. `champion` señala la versión que se usa en
producción; para promover otra versión o volver atrás basta con mover el alias, sin registrar
de nuevo ni cambiar el código que carga el modelo, que siempre pide `@champion`. Se puede
añadir un alias `challenger` para una candidata en evaluación. Las tres preguntas usan el mismo
esquema, sin diferencias entre ellas. En este proyecto, "producción" significa la versión a la
que apunta `champion` y cuya carga se comprobó contra el run de origen; no implica un servicio
desplegado. Para promover o revertir una versión:

```python
from mlflow import MlflowClient

MlflowClient().set_registered_model_alias("<modelo>", "champion", "<versión>")
```

Limitación conocida: en las hijas de la búsqueda de P2, `git_commit` está fijo en el código
(`32fb353`) y `register_p2` lee `data_version` de una etiqueta que la búsqueda no registra, así
que el linaje de P2 en el Registry muestra ese commit y deja `data_version` vacío.

## Pruebas y chequeos

Las pruebas viven en `tests/` y se corren con `make test`. Las capturas de evidencia se guardan
en `reports/figures/evidencia/`.

### Reproducibilidad del entrenamiento

#### P1

Responsable: [@MelizaMartinez98](https://github.com/MelizaMartinez98).

_Pendiente._

#### P2

Responsable: [@Eduardo-Valencia96](https://github.com/Eduardo-Valencia96).

_Pendiente._

#### P3

Responsable: [@LorenitaTL](https://github.com/LorenitaTL).

_Pendiente._

### Validación de datos con DeepChecks

Responsable: [@MelizaMartinez98](https://github.com/MelizaMartinez98).

_Pendiente._

### Pruebas de P1

Responsable: [@MelizaMartinez98](https://github.com/MelizaMartinez98).

_Pendiente._

### Pruebas de P2

Responsables: [@clsrivas](https://github.com/clsrivas) y
[@Eduardo-Valencia96](https://github.com/Eduardo-Valencia96).

_Pendiente._

### Seguridad

Responsable: [@clsrivas](https://github.com/clsrivas).

_Pendiente._

### Cobertura

Responsable: [@LorenitaTL](https://github.com/LorenitaTL).

_Pendiente._

## Buenas prácticas

- **Experimento único y runs con nombre descriptivo:** P1, P2 y P3 registran en
  `mibici-equipo02`, y cada run indica qué se probó y su versión, por ejemplo
  `baseline_semanal_v1`.
- **Parámetros de reproducción:** cada run registra fechas de partición, versión de los datos
  (`data_version`), semilla e hiperparámetros, para poder repetirlo.
- **Métricas con el conjunto en el nombre:** `mae_val`, `mae_test`, `test_recall`, etc., para
  no confundir validación con prueba.
- **Sin credenciales ni datos personales:** las credenciales van en `.env`, que no se versiona.
  La ingesta descarta `Usuario_Id`, `Genero` y `Año_de_nacimiento` de los CSV fuente, así que
  ningún dato personal llega a los modelos, al repositorio ni a MLflow.
- **Configuración en `config.py`:** rutas, fechas de corte y semillas se definen en
  `mi_bici/config.py` y se importan desde ahí, en lugar de repetirse en cada script.
- **Dependencias con versión fija:** `requirements.txt` fija la versión exacta de cada
  paquete (`paquete==x.y.z`).
- **`.gitignore`:** excluye datos (`data/`, `*.parquet`, CSV fuente), modelos serializados,
  `mlflow.db`, `mlartifacts/` y archivos de credenciales.
- **Etiquetas de trazabilidad:** cada run registra `team`, `question` (`P1`, `P2` o `P3`) y
  `git_commit` mediante `log_traceability()` de `mi_bici/modeling/tracking.py`, para ligarlo con
  la pregunta que atiende y el código que lo produjo.
- **Notebooks solo para exploración:** los notebooks siguen la convención
  `1.0-iniciales-descripcion.ipynb` (`1.0-evg-feature_engineering_p2`,
  `1.0-lt-perfiles-estaciones`), y la lógica final vive en el módulo `mi_bici/`.
- **Cambios por Pull Request revisado:** `main` está protegida; todo cambio entra por PR con al
  menos una aprobación y conversaciones resueltas (ver [Git workflow](#git-workflow)).

## Project Organization

```
├── LICENSE
├── Makefile           <- Comandos de entorno, lint, pruebas e ingesta
├── README.md
├── data               <- Excluida de Git
│   ├── external       <- Catálogo original de estaciones
│   ├── processed      <- Viajes normalizados (Parquet) y datasets de P1 y P2
│   └── raw            <- CSV mensuales originales
├── docs               <- Proyecto mkdocs y documentación de P1
├── models             <- Modelos locales de P2 (excluidos de Git)
├── notebooks          <- Exploración: 1.0-evg-feature_engineering_p2, 1.0-lt-perfiles-estaciones
├── pyproject.toml     <- Metadatos del paquete y configuración de ruff
├── references         <- ML Canvas, arquitectura, manifiesto de datos y reporte de calidad
├── reports
│   └── figures
├── requirements.txt   <- Dependencias con versión fija
├── tests              <- Pruebas de pytest
└── mi_bici            <- Código del proyecto
    ├── config.py          <- Rutas, fechas, semilla y configuración de MLflow
    ├── dataset.py         <- Ingesta, manifiesto y verificación de datos
    ├── data_quality.py    <- Reporte de calidad de datos
    ├── features.py        <- Funciones de features comunes a P1, P2 y P3
    ├── features_p1.py     <- Dataset estación-hora de P1
    ├── features_p2.py     <- Dataset estación-hora de P2
    ├── features_p3.py     <- Perfiles de uso por estación de P3
    ├── plots.py
    └── modeling
        ├── tracking.py        <- Experimento y trazabilidad comunes de MLflow
        ├── tracking_p1.py     <- Parámetros de partición de P1
        ├── splits_p1.py       <- Particiones cronológicas de P1
        ├── baseline_p1.py     <- Baseline estacional de P1
        ├── train_p1.py        <- Modelo de P1
        ├── evaluate_p1.py     <- Evaluación final de P1
        ├── baseline_p2.py     <- Baseline de persistencia de P2
        ├── train_p2.py        <- Modelo y ablaciones de P2
        ├── train_p3.py        <- Adaptador train, baseline y candidatos de P3
        ├── register_p3.py     <- Registro del modelo de P3 con alias champion
        └── predict.py         <- Plantilla de inferencia
```

--------
