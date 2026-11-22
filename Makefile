#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mi-bici-equipo-2
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
VALIDATION_ENV = .venv-validation
VALIDATION_PYTHON = $(VALIDATION_ENV)/bin/python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(MAKE) setup-hooks


## Install Locust for the load tests (on top of requirements.txt)
.PHONY: requirements-load
requirements-load:
	$(PYTHON_INTERPRETER) -m pip install -r requirements-load.txt


## Install pre-commit git hooks
.PHONY: setup-hooks
setup-hooks:
	pre-commit install




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Run tests with a coverage report
.PHONY: coverage
coverage:
	python -m pytest tests --cov=mi_bici --cov-report=term-missing


## Run security checks (use `make -k security` to see both reports)
.PHONY: security
security: security-code security-deps

## Analyze the code with bandit
.PHONY: security-code
security-code:
	bandit -c pyproject.toml -r mi_bici

# Accepted risk in the validation environment only: setuptools 80.10.2 has these two
# vulnerabilities, but DeepChecks 0.19.1 needs pkg_resources, removed in setuptools 81.
# That environment only validates data (make validate-data); it does not train or serve
# models. Remove these ignores once DeepChecks works without pkg_resources.
VALIDATION_IGNORED_VULNS = --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w

## Audit the dependencies with pip-audit (each file separately: their pins differ)
.PHONY: security-deps
security-deps:
	status=0; \
	pip-audit -r requirements.txt || status=1; \
	pip-audit -r requirements-load.txt || status=1; \
	pip-audit -r requirements-validation.txt $(VALIDATION_IGNORED_VULNS) || status=1; \
	exit $$status


## Create the DeepChecks environment (separate: it needs older numpy, scikit-learn and pandas)
.PHONY: create_validation_environment
create_validation_environment:
	uv venv $(VALIDATION_ENV) --python $(PYTHON_VERSION)
	uv pip install --python $(VALIDATION_ENV) -r requirements-validation.txt

## Validate the data with DeepChecks
.PHONY: validate-data
validate-data:
	$(VALIDATION_PYTHON) -m mi_bici.data_validation

## Run the DeepChecks tests
.PHONY: test-validation
test-validation:
	$(VALIDATION_PYTHON) -m pytest tests/test_data_validation.py


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	uv pip install --python .venv pip
	@echo ">>> New virtualenv created. Activate with:\nsource .venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m mi_bici.dataset main


## P2: build the June A/B set and the Evidently drift and quality report
.PHONY: drift-p2
drift-p2:
	$(PYTHON_INTERPRETER) -m mi_bici.monitoring.prepare_p2 --input data/processed/features_p2.parquet
	$(PYTHON_INTERPRETER) -m mi_bici.monitoring.drift_p2


#################################################################################
# API CONTAINER                                                                 #
#################################################################################

# Podman by default; use `make docker-run CONTAINER_ENGINE=docker` for Docker
CONTAINER_ENGINE ?= podman
IMAGE_NAME ?= mibici-api
CONTAINER_NAME ?= mibici-api
API_PORT ?= 8000

# The model reaches the container through the local MLflow store: mlflow.db is mounted at
# /app/mlflow.db and mlartifacts/ at the same absolute path it has on the host, because
# mlflow.db stores absolute artifact paths (file:///.../mlartifacts).
MLFLOW_DB ?= $(CURDIR)/mlflow.db
MLFLOW_ARTIFACTS ?= $(CURDIR)/mlartifacts
CONTAINER_MLFLOW_URI ?= sqlite:////app/mlflow.db

## Build the API image
.PHONY: docker-build
docker-build:
	$(CONTAINER_ENGINE) build -t $(IMAGE_NAME) .

## Run the API container in the background (model from the local MLflow store)
.PHONY: docker-run
docker-run:
	@if [ -f "$(MLFLOW_DB)" ] && [ -d "$(MLFLOW_ARTIFACTS)" ]; then \
		MOUNTS='-v "$(MLFLOW_DB):/app/mlflow.db" -v "$(MLFLOW_ARTIFACTS):$(MLFLOW_ARTIFACTS)"'; \
	else \
		echo "Warning: $(MLFLOW_DB) or $(MLFLOW_ARTIFACTS) not found; the API starts without a model."; \
		MOUNTS=''; \
	fi; \
	CMD="$(CONTAINER_ENGINE) run -d --rm --name $(CONTAINER_NAME) -p $(API_PORT):8000 \
		$$MOUNTS \
		-e MLFLOW_TRACKING_URI=$(CONTAINER_MLFLOW_URI) \
		-e MODEL_NAME -e MODEL_ALIAS -e MODEL_VERSION \
		$(IMAGE_NAME)"; \
	echo $$CMD; eval $$CMD
	@echo "API on http://localhost:$(API_PORT) (logs: make docker-logs, stop: make docker-stop)"

## Show the API container logs
.PHONY: docker-logs
docker-logs:
	$(CONTAINER_ENGINE) logs -f $(CONTAINER_NAME)

## Stop the API container
.PHONY: docker-stop
docker-stop:
	$(CONTAINER_ENGINE) stop $(CONTAINER_NAME)


#################################################################################
# SYSTEM: DOCKER COMPOSE, LOAD TESTS AND FINOPS                                 #
#################################################################################

# Docker Compose by default; use `make compose-up COMPOSE=podman-compose` for Podman.
# The services, ports and output files are the "Contrato de operación" in the README.
COMPOSE ?= docker compose
STATS_ENGINE ?= docker

## Start MLflow, the seed model and the API with one command (waits until healthy)
.PHONY: compose-up
compose-up:
	$(COMPOSE) up -d --build --wait
	@echo "API on http://localhost:$(API_PORT), MLflow on http://localhost:$${MLFLOW_PORT:-5001}"

## Stop the system (add COMPOSE_DOWN_FLAGS=-v to also delete the MLflow volume)
.PHONY: compose-down
compose-down:
	$(COMPOSE) down $(COMPOSE_DOWN_FLAGS)

## Follow the logs of every service
.PHONY: compose-logs
compose-logs:
	$(COMPOSE) logs -f

# Load test scenario: make load-test ESCENARIO=u50 USERS=50 SPAWN_RATE=5 DURATION=3m
ESCENARIO ?= u10
USERS ?= 10
SPAWN_RATE ?= 2
DURATION ?= 3m
STATS_INTERVAL ?= 2
LOAD_HOST ?= http://127.0.0.1:$(API_PORT)
LOAD_DIR = reports/load_test/$(ESCENARIO)
STATS_FILE = $(LOAD_DIR)/stats.jsonl

# One JSON line per container and sample: {"ts": "<UTC ISO>", "stats": <docker stats JSON>}.
# Only the containers of this Compose project are sampled.
STATS_LOOP = while true; do \
		ts=$$(date -u +%Y-%m-%dT%H:%M:%SZ); \
		$(STATS_ENGINE) stats --no-stream --format "{\"ts\":\"$$ts\",\"stats\":{{json .}}}" \
			$$($(COMPOSE) ps -q) >> $(STATS_FILE); \
		sleep $(STATS_INTERVAL); \
	done

## Record CPU and memory of the system containers until Ctrl+C (reports/load_test/<ESCENARIO>/stats.jsonl)
.PHONY: stats
stats:
	@mkdir -p $(LOAD_DIR)
	@echo "Recording $(STATS_ENGINE) stats in $(STATS_FILE) every $(STATS_INTERVAL) s (Ctrl+C to stop)"
	@$(STATS_LOOP)

## Run a headless Locust scenario against the API and record docker stats at the same time
.PHONY: load-test
load-test:
	@test -f load_tests/locustfile.py || { echo "load_tests/locustfile.py does not exist yet (M1)"; exit 1; }
	@mkdir -p $(LOAD_DIR)
	@rm -f $(STATS_FILE)
	@( $(STATS_LOOP) ) & stats_pid=$$!; \
	$(PYTHON_INTERPRETER) -m locust -f load_tests/locustfile.py --headless --only-summary \
		-u $(USERS) -r $(SPAWN_RATE) -t $(DURATION) --host $(LOAD_HOST) \
		--csv $(LOAD_DIR)/locust --csv-full-history; \
	status=$$?; kill $$stats_pid 2>/dev/null; \
	echo "Results in $(LOAD_DIR)/"; exit $$status

## Build the load test charts and reports/load_test/resumen.csv
.PHONY: load-report
load-report:
	@test -f mi_bici/operations/load_report.py || { echo "mi_bici/operations/load_report.py does not exist yet (M2)"; exit 1; }
	$(PYTHON_INTERPRETER) -m mi_bici.operations.load_report

## Estimate the cost per inference (reports/finops/costos.csv)
.PHONY: finops
finops:
	@test -f mi_bici/operations/finops.py || { echo "mi_bici/operations/finops.py does not exist yet (E2)"; exit 1; }
	$(PYTHON_INTERPRETER) -m mi_bici.operations.finops


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
