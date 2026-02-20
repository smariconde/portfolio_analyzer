from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_analyzer.core.settings import CACHE_DIR, CHARTS_DIR, ensure_directories


def test_ensure_directories_creates_required_paths() -> None:
    ensure_directories()
    assert CACHE_DIR.exists()
    assert CHARTS_DIR.exists()
