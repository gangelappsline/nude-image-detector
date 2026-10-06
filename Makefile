# Nude Image Detector - tareas habituales
#
#   make install   instala dependencias en .venv
#   make dev       servidor de desarrollo con recarga
#   make serve     gunicorn (como en producción)
#   make test      suite completa
#   make fast      suite sin el modelo real
#   make lint      ruff
#   make docker    imagen + contenedor

VENV   ?= .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
PORT   ?= 8000
ENGINE ?= nudenet

.DEFAULT_GOAL := help

.PHONY: help install install-dev dev serve test fast lint format docker docker-run smoke clean

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/python:
	python3 -m venv $(VENV)

install: $(VENV)/bin/python ## Crea el venv e instala dependencias de runtime
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: $(VENV)/bin/python ## Instala también las dependencias de desarrollo
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

dev: install-dev ## Servidor de desarrollo (recarga automática, logs en texto)
	NID_DEBUG=true NID_LOG_JSON=false NID_PORT=$(PORT) $(PY) run.py

serve: install ## Gunicorn, igual que en producción
	NID_PORT=$(PORT) $(VENV)/bin/gunicorn -c gunicorn.conf.py wsgi:app

test: install-dev ## Suite completa (incluye el modelo ONNX real)
	$(PY) -m pytest

fast: install-dev ## Suite sin cargar el modelo real
	$(PY) -m pytest -m "not slow"

lint: install-dev ## Analiza el código con ruff
	$(VENV)/bin/ruff check app tests
	$(VENV)/bin/ruff format --check app tests || true

format: install-dev ## Formatea el código
	$(VENV)/bin/ruff format app tests
	$(VENV)/bin/ruff check --fix app tests

docker: ## Construye la imagen
	docker build -t nude-image-detector:latest .

docker-run: ## Arranca el contenedor en el puerto PORT
	docker run --rm -p $(PORT):8000 --env-file .env nude-image-detector:latest

smoke: ## Comprueba /health, /ready y analiza una imagen generada al vuelo
	$(PY) scripts/smoke_test.py --base-url http://127.0.0.1:$(PORT)

clean: ## Borra caches y artefactos
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +
