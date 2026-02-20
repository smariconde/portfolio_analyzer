"""Deprecated script kept for backwards compatibility."""

from pathlib import Path


if __name__ == "__main__":
    print("`sharpe.py` is deprecated and archived.")
    print("Use `python -m portfolio_analyzer.cli run sortino` instead.")
    print(f"Archived source: {Path('scripts/legacy_archive/sharpe.py').resolve()}")
