from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from io import StringIO
import os
from typing import Any

import pandas as pd
import requests


LOCAL_AS_OF_DATE = date(2026, 1, 1)

LOCAL_COUNTRY_RISK_PREMIUM = {
    "US": 0.00,
    "UK": 0.01,
    "EU": 0.01,
    "JAPAN": 0.01,
    "BRAZIL": 0.03,
    "ARGENTINA": 0.07,
    "MEXICO": 0.025,
    "INDIA": 0.03,
    "CHINA": 0.03,
}

LOCAL_LONG_RUN_NOMINAL_GROWTH = {
    "US": 0.03,
    "UK": 0.03,
    "EU": 0.03,
    "JAPAN": 0.02,
    "BRAZIL": 0.045,
    "ARGENTINA": 0.06,
    "MEXICO": 0.04,
    "INDIA": 0.05,
    "CHINA": 0.045,
}


@dataclass(frozen=True)
class MarketAssumptions:
    risk_free_rate: float
    mature_erp: float
    country_risk_premium: float
    long_run_nominal_growth: float
    source: str
    as_of_date: str
    staleness_days: int


def _today_utc() -> date:
    return datetime.now(UTC).date()


def _to_date(value: str | None) -> date:
    if not value:
        return _today_utc()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return _today_utc()


def _online_fetch_enabled() -> bool:
    raw = str(os.environ.get("VALUATION_ONLINE_ASSUMPTIONS", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def _fetch_fred_gs10() -> tuple[float | None, str | None]:
    if not _online_fetch_enabled():
        return None, None
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10"
    try:
        response = requests.get(url, timeout=1.5)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
    except Exception:
        return None, None
    if frame.empty or "DATE" not in frame.columns or "GS10" not in frame.columns:
        return None, None
    frame["GS10"] = pd.to_numeric(frame["GS10"], errors="coerce")
    frame = frame.dropna(subset=["GS10"])
    if frame.empty:
        return None, None
    latest = frame.iloc[-1]
    return float(latest["GS10"]) / 100.0, str(latest["DATE"])


@lru_cache(maxsize=1)
def _fetch_damodaran_erp() -> tuple[float | None, str | None]:
    if not _online_fetch_enabled():
        return None, None
    # Historical implied ERP data (S&P 500) published by Damodaran.
    # File format can change; fallback to local values when parsing fails.
    url = "https://www.stern.nyu.edu/~adamodar/pc/datasets/histimpl.csv"
    try:
        response = requests.get(url, timeout=1.5)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
    except Exception:
        return None, None
    if frame.empty:
        return None, None
    lower_cols = {str(col).strip().lower(): col for col in frame.columns}
    date_col = lower_cols.get("date")
    erp_col = None
    for candidate in ("impliederp", "implied erp (fcfe)", "implied erp"):
        if candidate in lower_cols:
            erp_col = lower_cols[candidate]
            break
    if date_col is None or erp_col is None:
        return None, None
    frame[erp_col] = pd.to_numeric(frame[erp_col], errors="coerce")
    frame = frame.dropna(subset=[erp_col])
    if frame.empty:
        return None, None
    latest = frame.iloc[-1]
    return float(latest[erp_col]) / 100.0, str(latest[date_col])


@lru_cache(maxsize=64)
def get_market_assumptions(
    *,
    country: str,
    valuation_date: str | None = None,
) -> MarketAssumptions:
    country_key = (country or "US").strip().upper()
    target_date = _to_date(valuation_date)

    local_crp = LOCAL_COUNTRY_RISK_PREMIUM.get(country_key, 0.02)
    local_growth = LOCAL_LONG_RUN_NOMINAL_GROWTH.get(country_key, 0.03)

    rf = 0.04
    erp = 0.05
    as_of = LOCAL_AS_OF_DATE
    source = "local_fallback"

    online_rf, online_rf_date = _fetch_fred_gs10()
    online_erp, online_erp_date = _fetch_damodaran_erp()

    if online_rf is not None:
        rf = online_rf
        source = "online"
        if online_rf_date:
            try:
                as_of = datetime.strptime(online_rf_date, "%Y-%m-%d").date()
            except ValueError:
                pass
    if online_erp is not None:
        erp = online_erp
        source = "online"
        if online_erp_date:
            parsed = None
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
                try:
                    parsed = datetime.strptime(online_erp_date, fmt).date()
                    break
                except ValueError:
                    continue
            if parsed:
                as_of = parsed if parsed > as_of else as_of

    staleness_days = max((target_date - as_of).days, 0)
    return MarketAssumptions(
        risk_free_rate=rf,
        mature_erp=erp,
        country_risk_premium=local_crp,
        long_run_nominal_growth=local_growth,
        source=source,
        as_of_date=as_of.isoformat(),
        staleness_days=staleness_days,
    )


def market_assumptions_to_dict(assumptions: MarketAssumptions) -> dict[str, Any]:
    return {
        "risk_free_rate": assumptions.risk_free_rate,
        "mature_erp": assumptions.mature_erp,
        "country_risk_premium": assumptions.country_risk_premium,
        "long_run_nominal_growth": assumptions.long_run_nominal_growth,
        "source": assumptions.source,
        "as_of_date": assumptions.as_of_date,
        "staleness_days": assumptions.staleness_days,
    }
