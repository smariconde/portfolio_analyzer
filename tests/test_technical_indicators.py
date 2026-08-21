from __future__ import annotations

import pandas as pd

from ai_agent.tools import calculate_obv


def test_calculate_obv_tracks_up_down_and_flat_closes() -> None:
    prices = pd.DataFrame(
        {
            "close": [10.0, 12.0, 11.0, 11.0, 13.0],
            "volume": [100, 50, 40, 30, 20],
        }
    )

    result = calculate_obv(prices)

    assert result.tolist() == [0, 50, 10, 10, 30]
    assert prices["OBV"].tolist() == [0, 50, 10, 10, 30]
