# mi-bici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Mi Bici ML

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
| `make lint`     | `ruff format --check; ruff check`                                                   |
| `make format`   | `ruff check --fix; ruff format`                                                     |
| `make data`     | `python mi_bici/dataset.py`                                                         |
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

El modelo final corresponde al baseline con las 30 características.

Resultados obtenidos sobre los tres conjuntos:

| Conjunto | Recall | Precision | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.7505 | 0.5804 | 0.6546 | 0.8465 |
| Validation | 0.7391 | 0.5597 | 0.6370 | 0.8410 |
| Test | 0.7283 | 0.5524 | 0.6283 | 0.8364 |

El experimento se registra en MLflow bajo `P2-operational-risk`.

**Run final:** `e4385d55f1614c5989436bf066c9d791`

El entrenamiento puede reproducirse desde la raíz del repositorio mediante:

```bash
python -m mi_bici.modeling.train
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

Aunque la eliminación del desbalance actual produce un incremento marginal en Recall, el baseline presenta el mejor equilibrio general entre **Precision, F1 y ROC-AUC**. Por esta razón, se conserva el modelo con las 30 variables como modelo final de P2.

### Reproducibilidad de P2

Para reproducir el entrenamiento de P2 desde la raíz del repositorio:

```bash
python -m mi_bici.modeling.train
```

El script realiza la carga de las características procesadas, genera la partición temporal, calcula el umbral P80 a partir del conjunto de entrenamiento, entrena el pipeline de clasificación, evalúa el modelo en train, validation y test, y registra el experimento y sus artefactos en MLflow.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         mi_bici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mi_bici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mi_bici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
