# MiBici — Equipo 02

Proyecto de MLOps (TC5061) sobre los datos abiertos de viajes de MiBici, el sistema de
bicicletas públicas de Guadalajara. Con 18 meses de viajes (enero de 2025 a junio de 2026), el
equipo responde tres preguntas:

| Pregunta | Tipo | Modelo registrado |
| :--- | :--- | :--- |
| **P1.** ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | `mibici-equipo02-p1-demanda` |
| **P2.** ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Clasificación | `mibici-equipo02-p2-riesgo` |
| **P3.** ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | Clustering | `mibici-equipo02-p3-grupos` |

Los experimentos se registran en MLflow, en el experimento `mibici-equipo02`, y el modelo
elegido de cada pregunta está en el Model Registry con el alias `champion`.

## Reportes

- [P1. Predicción de demanda horaria](https://github.com/LorenitaTL/TC5061-Proyecto-Equipo02/blob/main/docs/p1_prediccion_demanda.md)
- [P1. Resultados de la Entrega 2](https://github.com/LorenitaTL/TC5061-Proyecto-Equipo02/blob/main/docs/p1_entrega2_resultados.md)
- [P2. Resultados de la Entrega 2](https://github.com/LorenitaTL/TC5061-Proyecto-Equipo02/blob/main/docs/p2_entrega2_resultados.md)
- [P3. Resultados de la Entrega 2](https://github.com/LorenitaTL/TC5061-Proyecto-Equipo02/blob/main/docs/p3_entrega2_resultados.md)

Para preparar el entorno y reproducir los resultados, ve a [Primeros pasos](getting-started.md).
