# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python-based finance toolkit with two main areas:
- Root scripts for data collection and analytics (for example `sortino.py`, `sharpe.py`, `bcra.py`, `indec_balanza_comercial.py`, `scrapper_diario.py`).
- `ai_agent/` for the Streamlit multi-page app and agent workflows (`Home.py`, `pages/`, `tools.py`, `backtester.py`).

Generated artifacts (charts, CSV, PDF) are typically written at the repository root, with cached market data in `cache/`.

## Build, Test, and Development Commands
Use Python 3.10+ and a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Run key workflows:
```bash
streamlit run ai_agent/Home.py                 # Launch AI agent UI
python sortino.py                              # Run Sortino sector analysis
python ai_agent/backtester.py --ticker AAPL    # Run backtest
python indec_balanza_comercial.py              # INDEC trade-balance chart
python bcra.py                                 # Interactive BCRA series analysis
```

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, snake_case for functions/variables, PascalCase for classes.
- Keep modules focused by domain (market data, analytics, agent UI).
- Prefer clear function names (`download_tickers_in_batches`) over abbreviations.
- If formatting is needed, use `black` locally before opening a PR.

## Testing Guidelines
There is no dedicated `tests/` suite yet; validation is primarily script-level and visual output checks.
- Add new tests under `tests/` using `pytest` when introducing non-trivial logic.
- For data scripts, include at least one reproducible smoke run command in the PR.
- For Streamlit changes, verify page load and key interactions locally with `streamlit run ai_agent/Home.py`.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries (for example: `Added csv output`, `Fixed sortino errors`).
- Keep commit subjects concise and action-oriented.
- In PRs, include: purpose, files changed, how to run/verify, and sample outputs (screenshots for UI, file names for reports).
- Link related issues/tasks and call out any API/data-source assumptions.
