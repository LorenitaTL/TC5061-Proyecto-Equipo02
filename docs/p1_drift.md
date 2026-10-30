# P1: diagnóstico de drift entre entrenamiento y prueba

## Alcance y reproducción

Ejecución desde el ambiente de validación con DeepChecks instalado:

    python -m mi_bici.data_validation

El comando genera `reports/p1_drift.html`.

Se comparan muestras aleatorias de 10,000 filas por conjunto, con semilla 42:

- Train: enero de 2025 a marzo de 2026; 4,028,792 filas disponibles.
- Test: junio de 2026; 289,902 filas disponibles.
- Validación (abril-mayo de 2026) queda fuera de esta comparación.

Las particiones provienen de `temporal_masks()`, utilizada por P1.

## Resultados

| Variable | Método | Drift score |
|---|---|---:|
| hour_of_day | V de Cramér | 0.0000 |
| day_of_week | V de Cramér | 0.0441 |
| is_weekend | V de Cramér | 0.0167 |
| lag_1h | Kolmogorov–Smirnov | 0.0425 |
| lag_24h | Kolmogorov–Smirnov | 0.0433 |
| lag_168h | Kolmogorov–Smirnov | 0.0323 |
| trip_count | Kolmogorov–Smirnov | 0.0359 |

Ningún predictor supera los umbrales exploratorios configurados:
0.2 para variables categóricas y 0.1 para numéricas.
Estos límites no están calibrados con impacto de negocio.
El objetivo `trip_count` se analiza sin condición de aprobación.

Los puntajes KS representan distancias entre distribuciones acumuladas,
no p-valores ni porcentajes de cambio en la demanda.

## Faltantes y límites

Los nulos de rezagos se conservan en las muestras y se excluyen del cálculo
de FeatureDrift mediante `ignore_na=True`:

| Rezago | Nulos en train | Nulos en test |
|---|---:|---:|
| lag_1h | 1 | 0 |
| lag_24h | 17 | 36 |
| lag_168h | 155 | 261 |

La validación previa del dataset completo confirmó que los nulos corresponden
a falta de historia inicial por estación.

El diagnóstico compara distribuciones marginales globales sobre muestras.
No comprueba estabilidad por estación, relaciones entre variables ni desempeño
predictivo. Los resultados no justifican por sí solos modificar datos o reentrenar.

## Pruebas y evidencia

Las pruebas sintéticas verifican distancia cero con datos idénticos y detección
de un cambio deliberado en un predictor y en el objetivo.
También verifican separación temporal, reproducibilidad de las muestras,
tamaño acotado y conservación del DataFrame original.

Captura: `reports/figures/evidencia/deepchecks_p1_drift.png`.