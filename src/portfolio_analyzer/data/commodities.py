from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bcra import get_minorista_exchange_rate


@dataclass
class CommoditiesResult:
    dataframe: pd.DataFrame
    correlation: float
    spread_pct_latest: float
    spread_pct_mean: float


def _load_chicago(cache_file: Path, cache_days: int = 1) -> pd.Series:
    if cache_file.exists():
        file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (datetime.now() - file_mod_time) < timedelta(days=cache_days):
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze("columns")
            return cached

    chicago = yf.download("ZS=F", period="5y", progress=False)["Close"]
    if chicago.empty:
        raise RuntimeError("Unable to download Chicago soybean data.")
    chicago.to_csv(cache_file)
    return chicago


def fetch_soybean_spread(
    start_date: str = "2024-01-01",
    cache_file: str = "cache/chicago_data.csv",
) -> CommoditiesResult:
    chicago_data = _load_chicago(Path(cache_file))
    chicago_usd_ton = chicago_data * 0.3674

    tables = pd.read_html("https://www.consiagro.com.ar/files/bd_pizarra_ros_historico.php")
    if not tables:
        raise RuntimeError("Rosario source returned no tables.")

    rosario_table = tables[0]
    if rosario_table.empty or rosario_table.shape[1] < 6:
        raise RuntimeError("Rosario source table is empty or malformed.")

    rosario_table.columns = ["Fecha", "Trigo", "Maiz", "Sorgo", "Soja_ARS", "Girasol"]
    rosario_table["Fecha"] = pd.to_datetime(rosario_table["Fecha"], format="%Y-%m-%d")
    rosario_table.set_index("Fecha", inplace=True)

    rosario_ars = (
        pd.to_numeric(
            rosario_table["Soja_ARS"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )
        .dropna()
        .sort_index()
        / 100
    )

    dolar = get_minorista_exchange_rate(
        fecha_desde=rosario_ars.index.min().strftime("%Y-%m-%d"),
        fecha_hasta=rosario_ars.index.max().strftime("%Y-%m-%d"),
    )
    if not dolar:
        raise RuntimeError("Unable to download BCRA FX data.")

    dolar_df = pd.DataFrame(dolar)
    dolar_df["fecha"] = pd.to_datetime(dolar_df["fecha"])
    dolar_df.set_index("fecha", inplace=True)

    rosario_usd = (
        rosario_ars / dolar_df["valor"].reindex(rosario_ars.index, method="ffill")
    ).dropna()

    start_dt = pd.Timestamp(start_date)
    chicago_daily = chicago_usd_ton[chicago_usd_ton.index >= start_dt].squeeze()
    rosario_daily = rosario_usd[rosario_usd.index >= start_dt].squeeze()

    combined = pd.DataFrame({"Chicago_USD_ton": chicago_daily})
    combined["Rosario_USD_ton"] = rosario_daily.reindex(combined.index, method="ffill")
    combined.dropna(inplace=True)

    if combined.empty:
        raise RuntimeError("No aligned soybean data for selected period.")

    combined["Spread_USD"] = combined["Chicago_USD_ton"] - combined["Rosario_USD_ton"]
    combined["Spread_Pct"] = (combined["Spread_USD"] / combined["Chicago_USD_ton"]) * 100

    correlation = float(combined["Chicago_USD_ton"].corr(combined["Rosario_USD_ton"]))
    spread_latest = float(combined["Spread_Pct"].iloc[-1])
    spread_mean = float(combined["Spread_Pct"].mean())

    return CommoditiesResult(
        dataframe=combined,
        correlation=correlation,
        spread_pct_latest=spread_latest,
        spread_pct_mean=spread_mean,
    )


def save_default_charts(df: pd.DataFrame, prices_path: str, spread_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(df.index, df["Chicago_USD_ton"], label="Chicago", color="#0077b6")
    ax.plot(df.index, df["Rosario_USD_ton"], label="Rosario", color="#d62828")
    ax.set_title("Soja diaria en USD/ton")
    ax.set_ylabel("USD/ton")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(prices_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ema20 = df["Spread_Pct"].ewm(span=20, adjust=False).mean()
    ax.plot(df.index, df["Spread_Pct"], label="Spread %", color="#2c3e50", alpha=0.6)
    ax.plot(df.index, ema20, label="EMA 20", color="#e74c3c")
    ax.axhline(df["Spread_Pct"].mean(), color="black", linestyle="--", linewidth=1)
    ax.set_title("Brecha porcentual Rosario vs Chicago")
    ax.set_ylabel("Spread %")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(spread_path)
    plt.close(fig)
