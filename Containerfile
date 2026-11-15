# Imagen de la API REST de MiBici (P2: riesgo de desbalance).
#
#   podman build -t mibici-api .
#   podman run --rm -p 8000:8000 mibici-api
#
# Sin MLflow la API arranca igual y /health reporta model_loaded: false.
# Cómo llega el modelo al contenedor: subsección «Podman/Docker» del README.

FROM docker.io/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# Dependencias primero (sin la línea `-e .`) para que la capa se reutilice
# mientras requirements.txt no cambie.
COPY requirements.txt .
RUN grep -v '^-e ' requirements.txt > /tmp/requirements-deps.txt \
    && pip install -r /tmp/requirements-deps.txt \
    && rm /tmp/requirements-deps.txt

# Paquete del proyecto. Se instala en modo editable, igual que `make requirements`,
# para que PROJ_ROOT (mi_bici/config.py) sea /app.
COPY pyproject.toml README.md LICENSE ./
COPY mi_bici/ mi_bici/
RUN pip install --no-deps -e .

# Usuario sin privilegios
RUN useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "mi_bici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
