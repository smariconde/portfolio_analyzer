from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_analyzer.analytics.sortino import _to_ranked_df


def test_to_ranked_df_filters_to_cedears_and_sorts_by_min_score() -> None:
    combined = pd.DataFrame(
        {
            "2 Years": [2.0, 1.0, 3.0, 0.5],
            "5 Years": [1.5, 2.5, 0.7, 0.6],
            "Last Price": [100, 200, 300, 400],
            "Last Date": ["2026-02-01"] * 4,
            "Sector": ["Tech", "Energy", "Health", "Finance"],
        },
        index=["AAA", "BBB", "CCC", "DDD"],
    )
    ranked = _to_ranked_df(combined, cedears=["BBB", "AAA", "DDD"])
    assert list(ranked["Ticker"]) == ["AAA", "BBB", "DDD"]


def test_to_ranked_df_coerces_numeric_strings() -> None:
    combined = pd.DataFrame(
        {
            "2 Years": ["1.2", "3.4"],
            "5 Years": ["0.8", "1.1"],
            "Last Price": ["100.5", "200.1"],
            "Last Date": ["2026-02-01", "2026-02-01"],
            "Sector": ["Tech", "Tech"],
        },
        index=["AAA", "BBB"],
    )
    ranked = _to_ranked_df(combined, cedears=["AAA", "BBB"])
    assert ranked["score_min"].notna().all()
    assert ranked["score_avg"].notna().all()

