# Primeros pasos

Las instrucciones completas están en el
[README del repositorio](https://github.com/LorenitaTL/TC5061-Proyecto-Equipo02#readme). Este
resumen sirve para macOS o Linux con Python 3.12; los comandos equivalentes para Windows
también están en el README.

## 1. Preparar el entorno

```bash
git clone git@github.com:LorenitaTL/TC5061-Proyecto-Equipo02.git
cd TC5061-Proyecto-Equipo02
make create_environment
source .venv/bin/activate
make requirements
make test
```

Si usas Anaconda, corre `conda deactivate` antes de activar `.venv`.

## 2. Descargar y verificar los datos

```bash
python -m mi_bici.dataset ingest --start-month 2025-01 --end-month 2026-06
python -m mi_bici.dataset verify
```

`ingest` descarga los CSV mensuales de MiBici y los normaliza en Parquet, y `verify` compara
sus huellas SHA-256 contra `references/data_manifest.json`.

## 3. Entrenar y registrar los modelos

Para levantar el servidor de MLflow, entrenar el modelo elegido de cada pregunta y registrarlo
con el alias `champion`, sigue la sección
[Servidor de tracking y MLflow Project](https://github.com/LorenitaTL/TC5061-Proyecto-Equipo02#servidor-de-tracking-y-mlflow-project)
del README.
