"""Deprecated script kept for backwards compatibility."""

from pathlib import Path

if __name__ == "__main__":
    print("`optimizacion_genetico.py` is deprecated.")
    print("Use `optimal_portfolio.py` or `python -m portfolio_analyzer.cli run optimizer ...`.")
    print(f"Archived source: {Path('scripts/legacy_archive/optimizacion_genetico.py').resolve()}")
