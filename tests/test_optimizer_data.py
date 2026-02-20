from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_analyzer.analytics.optimizer import _extract_close_prices


def test_extract_close_prices_simple_columns_uses_adj_close() -> None:
    raw = pd.DataFrame(
        {
            "Adj Close": [100.0, 101.0],
            "Close": [99.0, 100.0],
            "Volume": [1, 2],
        }
    )
    out = _extract_close_prices(raw)
    assert list(out.iloc[:, 0]) == [100.0, 101.0]


def test_extract_close_prices_multiindex_level0() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("Adj Close", "AAPL"),
            ("Adj Close", "MSFT"),
            ("Close", "AAPL"),
        ]
    )
    raw = pd.DataFrame(
        [
            [100.0, 200.0, 99.0],
            [101.0, 201.0, 100.0],
        ],
        columns=columns,
    )
    out = _extract_close_prices(raw)
    assert list(out.columns) == ["AAPL", "MSFT"]
    assert float(out.loc[0, "AAPL"]) == 100.0
    assert float(out.loc[1, "MSFT"]) == 201.0


def test_extract_close_prices_multiindex_level1() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("AAPL", "Close"),
            ("MSFT", "Close"),
            ("AAPL", "Volume"),
        ]
    )
    raw = pd.DataFrame(
        [
            [99.0, 199.0, 10],
            [100.0, 200.0, 20],
        ],
        columns=columns,
    )
    out = _extract_close_prices(raw)
    assert list(out.columns) == ["AAPL", "MSFT"]
    assert float(out.loc[0, "AAPL"]) == 99.0
    assert float(out.loc[1, "MSFT"]) == 200.0


def test_extract_close_prices_raises_when_no_close_like_column() -> None:
    raw = pd.DataFrame({"Open": [1.0, 2.0], "High": [2.0, 3.0]})
    with pytest.raises(RuntimeError):
        _extract_close_prices(raw)

