from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    outlier_dates: tuple[pd.Timestamp, ...] = ()


SOYBEAN_EVENTS = (
    ("2024-03-18", "Cosecha gruesa 2024"),
    ("2024-07-11", "Blend 80/20"),
    ("2024-11-01", "Fin Blend"),
    ("2025-01-28", "DNU 38/2025"),
    ("2025-03-24", "Cosecha gruesa 2025"),
    ("2025-06-24", "Pico DJVE"),
    ("2025-07-28", "Retenciones 26%"),
    ("2025-09-22", "Retenciones 0% temporal"),
)


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


def _clean_isolated_price_outliers(
    prices: pd.Series,
    jump_ratio: float = 2.0,
    neighbor_ratio: float = 1.5,
) -> tuple[pd.Series, tuple[pd.Timestamp, ...]]:
    """Interpolate only isolated one-day errors, preserving genuine price trends.

    A point is deemed erroneous only when it differs by more than ``jump_ratio``
    from both adjacent observations while those adjacent observations remain
    within ``neighbor_ratio`` of each other. This catches source typos such as
    an added zero without clipping legitimate market moves.
    """
    cleaned = prices.astype(float).sort_index().copy()
    previous = cleaned.shift(1)
    following = cleaned.shift(-1)

    ratio_to_previous = cleaned / previous
    ratio_to_following = cleaned / following
    neighbor_ratio_value = following / previous

    def is_large_move(ratio: pd.Series, threshold: float) -> pd.Series:
        return ratio.gt(threshold) | ratio.lt(1 / threshold)

    isolated_outlier = (
        previous.gt(0)
        & following.gt(0)
        & is_large_move(ratio_to_previous, jump_ratio)
        & is_large_move(ratio_to_following, jump_ratio)
        & ~is_large_move(neighbor_ratio_value, neighbor_ratio)
    )
    outlier_dates = tuple(pd.Timestamp(value) for value in cleaned.index[isolated_outlier])
    if outlier_dates:
        cleaned.loc[list(outlier_dates)] = float("nan")
        cleaned = cleaned.interpolate(method="time", limit_area="inside")

    return cleaned, outlier_dates


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

    dolar_aligned = dolar_df["valor"].reindex(rosario_ars.index, method="ffill")
    rosario_usd = (rosario_ars / dolar_aligned).dropna()
    rosario_usd, outlier_dates = _clean_isolated_price_outliers(rosario_usd)
    if outlier_dates:
        rosario_ars = rosario_ars.copy()
        rosario_ars.loc[list(outlier_dates)] = float("nan")
        rosario_ars = rosario_ars.interpolate(method="time", limit_area="inside")

    start_dt = pd.Timestamp(start_date)
    chicago_daily = chicago_usd_ton[chicago_usd_ton.index >= start_dt].squeeze()
    rosario_daily = rosario_usd[rosario_usd.index >= start_dt].squeeze()

    combined = pd.DataFrame({"Chicago_USD_ton": chicago_daily})
    combined["Rosario_USD_ton"] = rosario_daily.reindex(combined.index, method="ffill")
    combined["Rosario_ARS_ton"] = rosario_ars.reindex(combined.index, method="ffill")
    combined["FX_ARS_USD"] = dolar_aligned.reindex(combined.index, method="ffill")
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
        outlier_dates=outlier_dates,
    )


def _format_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _price_figure(df: pd.DataFrame, correlation: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    chicago = df["Chicago_USD_ton"]
    rosario = df["Rosario_USD_ton"]
    latest_spread = float(df["Spread_USD"].iloc[-1])
    latest_spread_pct = float(df["Spread_Pct"].iloc[-1])
    latest_rosario_ars = float(df["Rosario_ARS_ton"].iloc[-1])
    latest_fx = float(df["FX_ARS_USD"].iloc[-1])

    ax.plot(df.index, chicago, label="Chicago CBOT (USD/ton)", color="#0077b6", linewidth=1.8)
    ax.plot(df.index, rosario, label="Rosario Pizarra (USD/ton)", color="#d62828", linewidth=1.8)
    ax.fill_between(df.index, chicago, rosario, alpha=0.1, color="gray")

    for event_date_raw, label in SOYBEAN_EVENTS:
        event_date = pd.Timestamp(event_date_raw)
        if df.index.min() <= event_date <= df.index.max():
            ax.axvline(event_date, color="gray", linewidth=0.8, alpha=0.45)
            ax.text(
                event_date,
                0.02,
                label,
                rotation=90,
                fontsize=6,
                color="#4a4a4a",
                ha="right",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                bbox={"boxstyle": "round,pad=0.25", "fc": "#f8f9fa", "ec": "#cccccc", "lw": 0.5},
            )

    ax.set_title("Precios diarios de soja: Chicago vs. Rosario", fontsize=15, fontweight="bold", loc="left", pad=20)
    ax.text(
        0.01,
        0.95,
        f"Correlacion: {correlation:.4f} | Brecha: USD {latest_spread:+.2f} ({latest_spread_pct:+.2f}%)\n"
        f"Rosario: ARS {latest_rosario_ars:,.0f}/tn | TC minorista: ARS {latest_fx:,.2f}/USD",
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "#f8f9fa", "alpha": 0.95},
    )
    ax.set_ylabel("USD por tonelada")
    ax.set_xlabel("Fecha")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="black", fontsize=9)
    _format_date_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _spread_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    spread = df["Spread_Pct"]
    ema20 = spread.ewm(span=20, adjust=False).mean()
    hist_mean = float(spread.mean())
    last_date = df.index[-1]
    last_spread = float(spread.iloc[-1])
    last_ema = float(ema20.iloc[-1])
    latest_rosario_ars = float(df["Rosario_ARS_ton"].iloc[-1])
    latest_fx = float(df["FX_ARS_USD"].iloc[-1])

    ax.axhspan(0, 20, color="#27ae60", alpha=0.08, label="Venta agresiva")
    ax.axhspan(20, 25, color="#f39c12", alpha=0.08, label="Venta selectiva")
    ax.axhspan(25, max(40, float(spread.max()) + 2), color="#c0392b", alpha=0.08, label="Acumular")
    ax.plot(df.index, spread, color="#2c3e50", linewidth=1.2, alpha=0.7, label="Brecha diaria")
    ax.plot(df.index, ema20, color="#e74c3c", linewidth=2.2, label="EMA 20 dias")
    ax.axhline(hist_mean, color="black", linestyle="--", linewidth=1.4, label=f"Promedio historico: {hist_mean:+.2f}%")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)
    ax.scatter([last_date], [last_spread], color="#e74c3c", s=24, zorder=10)
    ax.annotate(
        f"{last_spread:+.2f}%",
        xy=(last_date, last_spread),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color="#e74c3c",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#e74c3c"},
    )

    ax.set_title("Brecha soja: Rosario vs. Chicago", fontsize=15, fontweight="bold", loc="left", pad=20)
    ax.text(
        0.01,
        0.95,
        f"Ultima sesion: {last_date:%d %b %Y} | Brecha: {last_spread:+.2f}% | EMA20: {last_ema:+.2f}%\n"
        f"Rosario: ARS {latest_rosario_ars:,.0f}/tn | TC minorista: ARS {latest_fx:,.2f}/USD",
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "#f8f9fa", "alpha": 0.95},
    )
    ax.set_ylabel("Brecha (%)")
    ax.set_xlabel("Fecha")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="black", fontsize=8)
    _format_date_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def generate_soybean_report_pdf(result: CommoditiesResult, start_date: str) -> bytes:
    """Create the detailed, print-ready soybean report formerly produced by the script."""
    df = result.dataframe
    latest = df.iloc[-1]
    report = BytesIO()

    with PdfPages(report) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.88, "Informe de soja: Chicago vs. Rosario", fontsize=24, fontweight="bold")
        fig.text(0.08, 0.83, f"Periodo analizado: {start_date} a {df.index[-1]:%Y-%m-%d}", fontsize=11, color="#555555")
        summary = (
            f"Correlacion Pearson: {result.correlation:.4f}\n"
            f"Chicago: USD {float(latest['Chicago_USD_ton']):,.2f} por ton\n"
            f"Rosario: USD {float(latest['Rosario_USD_ton']):,.2f} por ton\n"
            f"Rosario: ARS {float(latest['Rosario_ARS_ton']):,.0f} por ton\n"
            f"Tipo de cambio minorista BCRA: ARS {float(latest['FX_ARS_USD']):,.2f} por USD\n"
            f"Brecha actual: {result.spread_pct_latest:+.2f}%\n"
            f"Brecha promedio: {result.spread_pct_mean:+.2f}%"
        )
        fig.text(0.08, 0.58, summary, fontsize=14, linespacing=1.7, va="top")
        fig.text(
            0.08,
            0.16,
            "Fuentes: futuros de soja CBOT (Yahoo Finance), pizarra Rosario (Consiagro) y tipo de cambio minorista BCRA.",
            fontsize=9,
            color="#666666",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for figure in (_price_figure(df, result.correlation), _spread_figure(df)):
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    return report.getvalue()


def save_default_charts(df: pd.DataFrame, prices_path: str, spread_path: str) -> None:
    correlation = float(df["Chicago_USD_ton"].corr(df["Rosario_USD_ton"]))
    price_figure = _price_figure(df, correlation)
    price_figure.savefig(prices_path, dpi=300, bbox_inches="tight")
    plt.close(price_figure)

    spread_figure = _spread_figure(df)
    spread_figure.savefig(spread_path, dpi=300, bbox_inches="tight")
    plt.close(spread_figure)
