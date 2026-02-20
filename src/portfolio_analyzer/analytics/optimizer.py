from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models


@dataclass
class OptimizerResult:
    weights: pd.DataFrame
    sortino_ratio: float
    benchmark_sortino: float
    benchmark_sharpe: float


def _extract_close_prices(raw: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return a close-price matrix from yfinance output across schema variants."""
    if isinstance(raw, pd.Series):
        return raw.to_frame(name=raw.name or "price")

    if raw.empty:
        return pd.DataFrame()

    candidates = ("Adj Close", "Close", "adjclose", "close")

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))

        for field in candidates:
            if field in lvl0:
                out = raw[field]
                if isinstance(out, pd.Series):
                    return out.to_frame(name=out.name or "price")
                return out
            if field in lvl1:
                out = raw.xs(field, axis=1, level=1)
                if isinstance(out, pd.Series):
                    return out.to_frame(name=out.name or "price")
                return out
    else:
        for field in candidates:
            if field in raw.columns:
                out = raw[field]
                if isinstance(out, pd.Series):
                    return out.to_frame(name=out.name or "price")
                return out

    raise RuntimeError(
        "Could not find price column in yfinance response (expected Adj Close or Close)."
    )


def optimize_portfolio(
    tickers: list[str],
    portfolio_size: float,
    cash_percentage: float = 0.6,
    min_weight: float = 0.05,
    max_weight: float = 0.40,
    years: int = 5,
) -> OptimizerResult:
    if not tickers:
        raise ValueError("At least one ticker is required.")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    raw_prices = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = _extract_close_prices(raw_prices)
    if prices.empty:
        raise RuntimeError("No price data found for the selected tickers.")

    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        raise RuntimeError("No valid price columns remained after cleaning.")

    raw_sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    sp500_data = _extract_close_prices(raw_sp500).iloc[:, 0]

    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.sample_cov(prices)

    ef = EfficientFrontier(mu, cov)
    ef.add_constraint(lambda w: w >= min_weight)
    ef.add_constraint(lambda w: w <= max_weight)
    ef.max_quadratic_utility(risk_aversion=2, market_neutral=False)
    cleaned_weights = ef.clean_weights()

    total_investment = portfolio_size * (1 - cash_percentage)
    weights_df = pd.DataFrame.from_dict(cleaned_weights, orient="index", columns=["weight"])
    weights_df["investment"] = weights_df["weight"] * total_investment
    weights_df = weights_df[weights_df["investment"] > 0].sort_values("weight", ascending=False)

    weights = weights_df["weight"].copy()
    common_tickers = [ticker for ticker in prices.columns if ticker in weights.index]
    if not common_tickers:
        raise RuntimeError("No overlap between optimizer weights and downloaded prices.")

    weighted_returns = prices[common_tickers].pct_change().dropna().mul(
        weights.reindex(common_tickers), axis=1
    )
    portfolio_returns = weighted_returns.sum(axis=1)
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (
        float(portfolio_returns.mean() / downside_std * np.sqrt(252))
        if downside_std and not np.isnan(downside_std)
        else float("nan")
    )

    sp500_returns = sp500_data.pct_change().dropna()
    benchmark_return = sp500_returns.mean() * 252
    benchmark_vol = sp500_returns.std() * np.sqrt(252)
    benchmark_sharpe = float(benchmark_return / benchmark_vol)

    benchmark_downside = sp500_returns[sp500_returns < 0].std() * np.sqrt(252)
    benchmark_sortino = (
        float(benchmark_return / benchmark_downside)
        if benchmark_downside != 0
        else float("nan")
    )

    return OptimizerResult(
        weights=weights_df,
        sortino_ratio=sortino_ratio,
        benchmark_sortino=benchmark_sortino,
        benchmark_sharpe=benchmark_sharpe,
    )
