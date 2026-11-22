# Análisis FinOps

## Supuestos

- RPS por réplica: **10**
- Utilización considerada: **70%**
- Volumen diario de referencia: **1,000 inferencias**
- Días por mes: **30**
- Moneda: **USD**
- Modelo de precio: **On-Demand**

## Comparación de instancias

| provider | instancia | region | vcpu | memory_gb | precio_usd_h | rps | utilizacion | costo_inferencia_usd | costo_1k_usd | costo_mensual_usd | replicas_10_rps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AWS | t3.medium | us-east-1 | 2 | 4 | 0.04160000 | 10.0 | 0.7 | 0.00000165 | 0.00165079 | 0.04952381 | 2 |
| AWS | c6i.large | us-east-1 | 2 | 4 | 0.08500000 | 10.0 | 0.7 | 0.00000337 | 0.00337302 | 0.10119048 | 2 |
| GCP | e2-standard-2 | us-central1 | 2 | 8 | 0.06700000 | 10.0 | 0.7 | 0.00000266 | 0.00265873 | 0.07976190 | 2 |

## Fuentes

- **AWS t3.medium**: AWS EC2 On-Demand Pricing / AWS Prescriptive Guidance — https://aws.amazon.com/ec2/pricing/on-demand/
- **AWS c6i.large**: AWS Architecture Blog / EC2 pricing reference — https://aws.amazon.com/ec2/pricing/on-demand/
- **GCP e2-standard-2**: Google Cloud Compute Engine pricing reference — https://cloud.google.com/compute/all-pricing

## Interpretación

El costo por inferencia se obtiene dividiendo el costo horario entre la capacidad efectiva de inferencia. La capacidad efectiva considera el RPS medido y el porcentaje de utilización seleccionado.

Estos valores son estimaciones de cómputo y no incluyen almacenamiento, transferencia de datos, balanceadores, bases de datos, observabilidad ni impuestos.
