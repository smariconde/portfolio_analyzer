from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "cache"
CHARTS_DIR = PROJECT_ROOT / "charts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_SORTINO_CSV = OUTPUTS_DIR / "sortino" / "all_sortino_ratios.csv"
DEFAULT_CEDEARS_TXT = OUTPUTS_DIR / "sortino" / "cedears_selection.txt"


def ensure_directories() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "sortino").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "commodities").mkdir(parents=True, exist_ok=True)


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    return value
