# Ayni — atajos del pipeline.
# En Windows no hay make: usa `python -m ayni <comando>` dentro del venv.

PY      ?= python
VENV    ?= .venv
BIN     := $(VENV)/bin
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
endif
RUN     := $(BIN)/python -m ayni

.DEFAULT_GOAL := help

.PHONY: help install data download parquet check test lint fmt features train evaluate drift clean

help:
	@echo "Ayni — targets disponibles"
	@echo "  install    crea el venv e instala requirements.txt"
	@echo "  data       verifica que los CSV de Kaggle esten en data/raw"
	@echo "  download   descarga y descomprime el dataset de Kaggle"
	@echo "  parquet    convierte los CSV a Parquet en data/interim"
	@echo "  check      lista filas y columnas de cada Parquet"
	@echo "  test       pytest"
	@echo "  lint       ruff check"
	@echo "  fmt        ruff format"
	@echo "  clean      borra caches y artefactos generados"

install:
	python -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -r requirements.txt
	$(BIN)/python -m pip install -e . --no-deps
	$(BIN)/pre-commit install

data:
	$(RUN) data

download:
	$(RUN) download

parquet:
	$(RUN) parquet

check:
	$(RUN) check

features:
	$(RUN) features

train:
	$(RUN) train

evaluate:
	$(RUN) evaluate

drift:
	$(RUN) drift

test:
	$(BIN)/python -m pytest

lint:
	$(BIN)/ruff check src tests

fmt:
	$(BIN)/ruff format src tests

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data/interim/*.parquet
