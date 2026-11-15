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
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
