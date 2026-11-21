# Resumen de pruebas de carga

Las métricas de solicitudes corresponden únicamente a POST /predict con payload válido. La duración es el intervalo observado en el historial de Locust.

| escenario | usuarios | duracion_s | peticiones | fallas | tasa_error | rps | p50_ms | p95_ms | p99_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| u10 | 10 | 178 | 1104 | 0 | 0.0000 | 6.2148 | 15.0 | 50.0 | 68.0 |
| u50 | 50 | 177 | 4955 | 0 | 0.0000 | 27.8751 | 72.0 | 620.0 | 830.0 |
| u100 | 100 | 177 | 5253 | 0 | 0.0000 | 29.6759 | 1400.0 | 2500.0 | 5400.0 |
