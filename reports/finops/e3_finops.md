# Análisis FinOps E3

## Supuestos

- Estaciones de referencia: **374**
- Predicciones por estación: **24/día**
- Volumen de referencia: **8,976 inferencias/día**
- Escenario de estrés: **10x**
- Volumen de estrés: **89,760 inferencias/día**
- Días por mes: **30**
- Horas de operación mensual para costo 24/7: **730 h**
- Moneda: **USD**
- Capacidad utilizada: RPS medido en cada escenario de M3

## Resultados de rendimiento y recursos

| escenario | usuarios | rps | p95_ms | cpu_pct_prom | cpu_pct_p95 | mem_mib_prom | mem_mib_p95 | tasa_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| u10 | 10 | 6.2148 | 50.0000 | 15.6691 | 52.3475 | 223.8978 | 223.8000 | 0.0000 |
| u50 | 50 | 27.8751 | 620.0000 | 66.3031 | 94.8120 | 232.9311 | 243.6800 | 0.0000 |
| u100 | 100 | 29.6759 | 2500.0000 | 93.0524 | 107.2060 | 241.8711 | 253.1800 | 0.0000 |

## Análisis FinOps

El volumen de referencia corresponde a 374 estaciones con una predicción horaria por estación:

**374 × 24 = 8,976 inferencias/día.**

Para el escenario de estrés de 10x:

**8,976 × 10 = 89,760 inferencias/día.**

El RPS promedio requerido para procesar este volumen continuamente durante 24 horas es aproximadamente:

**89,760 / 86,400 = 1.0389 RPS.**

Las réplicas necesarias se calculan comparando este RPS objetivo con el RPS medido experimentalmente para cada escenario de carga.

| escenario | rps | target_rps_10x | replicas_10x | costo_inferencia_usd | costo_1000_usd | costo_mes_10x_usd | costo_24x7_mes_usd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| u10 | 6.214781 | 1.038889 | 1 | 0.00000186 | 0.001859 | 5.006902 | 30.368000 |
| u50 | 27.875052 | 1.038889 | 1 | 0.00000041 | 0.000415 | 1.116296 | 30.368000 |
| u100 | 29.675852 | 1.038889 | 1 | 0.00000039 | 0.000389 | 1.048556 | 30.368000 |

## Interpretación

El escenario u10 alcanza aproximadamente 6.21 RPS con un P95 de latencia de 50 ms.

El escenario u50 alcanza aproximadamente 27.88 RPS con un P95 de latencia de 620 ms.

Al aumentar la carga hasta u100, el throughput solo aumenta hasta aproximadamente 29.68 RPS, mientras que el P95 aumenta hasta 2,500 ms.

Esto indica que la API comienza a saturarse alrededor de 30 RPS bajo la configuración evaluada.

El consumo de CPU de la API aumenta de forma importante con la carga: el promedio pasa de 15.7% en u10 a 66.3% en u50 y 93.1% en u100.

El CPU p95 alcanza aproximadamente 107.2% en u100, lo que indica que la configuración actual se encuentra bajo presión de CPU al acercarse a 30 RPS.

El consumo de memoria de la API permanece relativamente estable, alrededor de 224–242 MiB promedio, por lo que CPU y latencia constituyen los principales indicadores de saturación de la API.

El contenedor de MLflow presenta un consumo de memoria cercano al límite configurado de 1 GiB. Esto representa un riesgo de capacidad que debe considerarse en futuras optimizaciones, aunque los datos disponibles no permiten atribuirle directamente la latencia observada en la API.

El volumen de 8,976 inferencias/día es un supuesto de dimensionamiento basado en 374 estaciones y una predicción horaria por estación; no representa un volumen oficial declarado por MiBici.

El cálculo de costo 24/7 representa el costo de mantener las réplicas encendidas durante 730 horas mensuales. El costo por inferencia representa una métrica de eficiencia basada en el RPS medido y no debe interpretarse como el costo de una instancia encendida continuamente.
