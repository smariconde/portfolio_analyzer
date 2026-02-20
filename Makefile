PYTHON := .venv/bin/python
PIP := .venv/bin/pip
CLI := .venv/bin/portfolio-analyzer

.PHONY: setup ui test lint check sortino optimizer commodities

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install pytest ruff

ui:
	$(CLI) ui

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests

check: lint test

sortino:
	$(CLI) run sortino

optimizer:
	$(CLI) run optimizer --tickers AAPL,MSFT,NVDA --portfolio-size 40000

commodities:
	$(CLI) run commodities --start-date 2024-01-01
