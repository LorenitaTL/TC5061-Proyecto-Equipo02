# Checklist – Entrega 1: Propuesta y base reproducible

**Fecha de entrega:** sábado 17 de octubre de 2026, 23:59

Marca cada punto con `[x]` cuando esté terminado. Los puntos marcados con ⚠️ necesitan
atención antes de lo demás.

---

## 0. Pendientes urgentes

- [x] Renombrar el repo en GitHub a `TC5061-Proyecto-Equipo02`
- [ ] ⚠️ Dar acceso al repo al profesor
- [ ] ⚠️ Proteger la rama `main` (Settings → Branches)
- [ ] Hacer merge del PR de pre-commit + Conventional Commits

---

## Parte A. Propuesta de solución (va en el PDF)

### 1. Formular el problema
- [ ] ML Canvas del proyecto completo
- [ ] P1 (regresión): qué se predice, granularidad, horizonte, relevancia, usuario y decisión
- [ ] P2 (clasificación): qué se predice, granularidad, horizonte, relevancia, usuario y decisión
- [ ] P2: definir y justificar cómo se aproxima el riesgo con viajes + capacidad de estación
- [ ] P3 (clustering): qué se agrupa, granularidad, relevancia, usuario y decisión

### 2. Definir los datos
- [ ] Rango de fechas elegido y justificado
- [ ] Fuente: enlaces a los archivos
- [ ] Tamaño real: número de meses, viajes y estaciones (cifras reales)
- [ ] Variables originales usadas y cómo se construye cada variable objetivo
- [ ] Condiciones de uso de MiBici y tratamiento de datos personales
- [ ] Estrategia de partición que evite usar información del futuro

### 3. Diseñar el pipeline de ML (para P1, P2 y P3)
- [ ] Etapas: limpieza → agregaciones → feature engineering → entrenamiento → evaluación
- [ ] Variables que se construirán y por qué son pertinentes

### 4. Modelos y métricas (para P1, P2 y P3)
- [ ] Baseline sencillo
- [ ] Al menos un modelo candidato
- [ ] Métricas justificadas según la tarea y la decisión

### 5. Arquitectura del sistema
- [ ] Diagrama: fuentes → ingesta → almacenamiento → procesamiento → entrenamiento → registro de experimentos
- [ ] Herramienta de cada componente, justificada con las lecturas de la semana

### 6. Experimento base en MLflow
- [ ] Experimento con número de equipo (ej. `mibici-equipo02`)
- [ ] Al menos un run por pregunta (P1, P2, P3) con nombre descriptivo
- [ ] Parámetros: rango de fechas, versión de los datos, semilla, modelo e hiperparámetros
- [ ] Métricas con el conjunto en el nombre (ej. `test_mae`, no `mae`)
- [ ] Al menos un artefacto por run (gráfica, mapa de clusters, etc.)
- [ ] Etiquetas: equipo, pregunta (`P1`/`P2`/`P3`) y commit de Git
- [ ] Los runs se ejecutan desde el código del repo, no solo desde un notebook
- [ ] No hay credenciales ni datos personales en parámetros, etiquetas o artefactos

### 7. Capturas de MLflow
- [ ] Lista de runs del experimento
- [ ] Detalle de un run con parámetros, métricas y etiquetas
- [ ] Al menos un artefacto registrado
- [ ] Comparación de dos o más runs

---

## Parte B. Repositorio del equipo

### 8. Estructura base
- [x] Repo generado con `ccds` e `include_code_scaffold = Yes`, subido como primer commit
- [x] Separación de carpetas de la plantilla (`data/`, `notebooks/`, `mi_bici/`, `models/`, `reports/figures/`)
- [ ] Rutas, fechas y semillas centralizadas en `mi_bici/config.py` (ya están las rutas; faltan fechas y semillas)
- [ ] Versiones fijas de las dependencias en `requirements.txt`
- [x] `.gitignore` ampliado:
  - [x] Datos (`/data/`)
  - [x] `.env`
  - [x] `mlruns/`
  - [x] `mlartifacts/`
  - [x] Modelos serializados (`models/*`, `*.pkl`, `*.joblib`, etc.)
- [ ] ML Canvas en `references/`
- [ ] Diagrama de arquitectura en `references/`

### 9. Flujo de trabajo Git/GitHub (documentado en el README)
- [ ] Rama principal protegida; todo entra por PR (GitHub Flow)
- [ ] Convención de nombres de ramas documentada
- [x] Convención de commits definida (Conventional Commits + hook de pre-commit)
- [ ] Integrante 1: abrió al menos un PR y revisó al menos uno
- [ ] Integrante 2: abrió al menos un PR y revisó al menos uno
- [ ] Integrante 3: abrió al menos un PR y revisó al menos uno
- [ ] Integrante 4: abrió al menos un PR y revisó al menos uno

### 10. Ingesta reproducible (`mi_bici/dataset.py`)
- [ ] Se ejecuta desde la terminal sin pasos manuales
- [ ] Descarga los meses elegidos y la nomenclatura de estaciones
- [ ] Normaliza nombres de columnas, codificación de caracteres y tipos de fecha
- [ ] Guarda los datos limpios en Parquet (u otro formato eficiente)
- [ ] Versiona los datos (ej. manifiesto en Git con número de registros y hash de cada archivo)
- [ ] La versión de los datos se puede registrar en MLflow

### 11. Feature engineering (`mi_bici/features.py`)
- [ ] Transformaciones del notebook pasadas a funciones reutilizables
- [ ] Docstring en cada función: qué hace y por qué es pertinente para su pregunta
- [ ] Sin fuga de información del futuro
- [ ] Al menos un notebook de exploración con nombre `1.0-iniciales-descripcion.ipynb`

### 12. README
- [ ] Propósito del proyecto
- [ ] Las tres preguntas
- [ ] Datos utilizados
- [ ] Estructura de carpetas
- [ ] Instalación (entorno + dependencias)
- [ ] Comando de ingesta
- [ ] Comando del experimento base
- [ ] Flujo de trabajo con Git
- [ ] Integrantes del equipo y sus roles
- [ ] Cómo se aplican las buenas prácticas obligatorias

---

## Entrega final

### PDF `TC5061_Propuesta_Proyecto_Equipo02.pdf`
- [ ] Enlace al repositorio
- [ ] ML Canvas y formulación de P1, P2 y P3
- [ ] Definición de los datos y partición
- [ ] Pipeline de ML
- [ ] Baselines, modelos candidatos y métricas por pregunta
- [ ] Diagrama de arquitectura con herramientas justificadas
- [ ] Resultados del experimento base
- [ ] Las cuatro capturas de MLflow (paso 7)
- [ ] Captura de la estructura del repositorio
- [ ] Referencias en formato APA 7
- [ ] Revisión de redacción y ortografía

### Verificación antes de entregar
- [ ] El repo se llama `TC5061-Proyecto-Equipo02` y el profesor tiene acceso
- [ ] En un clon limpio, siguiendo el README, se instalan las dependencias y corren la ingesta y el experimento sin pasos manuales
- [ ] No hay datos, `mlruns/`, modelos ni credenciales en el historial de Git
- [ ] El experimento tiene al menos un run por pregunta con parámetros, métricas, etiquetas y artefactos
- [ ] Los cuatro integrantes tienen commits, al menos un PR abierto y al menos una revisión
- [ ] Un integrante sube el PDF a Canvas
