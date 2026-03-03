# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python-based finance toolkit with three main areas:
- `src/portfolio_analyzer/` for the installable package:
  - `cli/` command-line entrypoint (`portfolio-analyzer`)
  - `analytics/` portfolio logic (sortino, optimizer, agent)
  - `data/` macro and commodities data loaders
  - `core/` shared settings and logging
- `ai_agent/` for the Streamlit multi-page app and agent workflows (`Home.py`, `pages/`, `tools.py`, `valuation_v2/`, `backtester.py`).
- `scripts/` for script-style workflows (for example `sortino.py`, `bcra.py`, `indec_balanza_comercial.py`, `optimal_portfolio.py`) and `scripts/legacy/` archives.

Generated artifacts (charts, CSV, PDF) are typically written to `outputs/`, `charts/`, or repository root, with cached market data in `cache/`.

## Build, Test, and Development Commands
Use Python 3.10+ and a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
Run key workflows (preferred via `Makefile`):
```bash
make setup                        # Install runtime + dev tooling
make ui                           # Launch Streamlit UI via CLI
make test                         # Run pytest suite
make lint                         # Run ruff on src/tests
make check                        # Lint + tests
```
Equivalent direct commands:
```bash
portfolio-analyzer ui
portfolio-analyzer run sortino
portfolio-analyzer run optimizer --tickers AAPL,MSFT,NVDA --portfolio-size 40000
portfolio-analyzer run commodities --start-date 2024-01-01
streamlit run ai_agent/Home.py
```

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, snake_case for functions/variables, PascalCase for classes.
- Keep modules focused by domain (market data, analytics, agent UI).
- Prefer clear function names (`download_tickers_in_batches`) over abbreviations.
- Use `ruff` checks from the `Makefile`; keep imports sorted and lines within configured limits.

## Testing Guidelines
The repository includes a `pytest` suite under `tests/`.
- Add or update tests in `tests/` when changing analytics, CLI, or data logic.
- Run `make test` (or `python -m pytest -q`) before opening a PR.
- For Streamlit changes, verify page load and key interactions locally with `make ui` or `streamlit run ai_agent/Home.py`.
- For script/data changes, include at least one reproducible smoke-run command in the PR description.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries (for example: `Added csv output`, `Fixed sortino errors`).
- Keep commit subjects concise and action-oriented.
- In PRs, include: purpose, files changed, how to run/verify, and sample outputs (screenshots for UI, file names for reports).
- Link related issues/tasks and call out any API/data-source assumptions.
