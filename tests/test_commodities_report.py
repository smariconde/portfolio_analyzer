from __future__ import annotations

import pandas as pd

from portfolio_analyzer.data.commodities import (
    CommoditiesResult,
    _clean_isolated_price_outliers,
    generate_soybean_report_pdf,
)


def test_clean_isolated_price_outlier_interpolates_a_single_bad_observation() -> None:
    index = pd.date_range("2026-03-15", periods=5, freq="D")
    prices = pd.Series([320.0, 328.0, 3336.0, 338.0, 340.0], index=index)

    cleaned, outlier_dates = _clean_isolated_price_outliers(prices)

    assert outlier_dates == (pd.Timestamp("2026-03-17"),)
    assert cleaned.loc[pd.Timestamp("2026-03-17")] == 333.0


def test_clean_isolated_price_outlier_preserves_normal_price_moves() -> None:
    index = pd.date_range("2026-03-15", periods=4, freq="D")
    prices = pd.Series([320.0, 328.0, 338.0, 340.0], index=index)

    cleaned, outlier_dates = _clean_isolated_price_outliers(prices)

    assert not outlier_dates
    assert cleaned.equals(prices)


def test_generate_soybean_report_pdf() -> None:
    index = pd.date_range("2025-01-01", periods=30, freq="D")
    chicago = pd.Series(range(400, 430), index=index, dtype=float)
    rosario = pd.Series(range(300, 330), index=index, dtype=float)
    dataframe = pd.DataFrame(
        {
            "Chicago_USD_ton": chicago,
            "Rosario_USD_ton": rosario,
            "Rosario_ARS_ton": rosario * 1_000,
            "FX_ARS_USD": 1_000.0,
            "Spread_USD": chicago - rosario,
            "Spread_Pct": ((chicago - rosario) / chicago) * 100,
        }
    )
    result = CommoditiesResult(
        dataframe=dataframe,
        correlation=1.0,
        spread_pct_latest=float(dataframe["Spread_Pct"].iloc[-1]),
        spread_pct_mean=float(dataframe["Spread_Pct"].mean()),
    )

    report = generate_soybean_report_pdf(result, "2025-01-01")

    assert report.startswith(b"%PDF")
    assert len(report) > 10_000
