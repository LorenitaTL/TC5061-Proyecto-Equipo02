# mi-bici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Mi Bici ML

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

