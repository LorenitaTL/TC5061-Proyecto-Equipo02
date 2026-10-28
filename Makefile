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
	bandit -r mi_bici

## Audit the dependencies with pip-audit (each file separately: their pins differ)
.PHONY: security-deps
security-deps:
	status=0; \
	pip-audit -r requirements.txt || status=1; \
	pip-audit -r requirements-validation.txt || status=1; \
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
