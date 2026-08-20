from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from matplotlib.backends.backend_pdf import PdfPages


@dataclass
class CommoditiesResult:
    dataframe: pd.DataFrame
    correlation: float
    spread_pct_latest: float
    spread_pct_mean: float
    spread_percentile: float = float("nan")
    decision: str = ""
    outlier_dates: tuple[pd.Timestamp, ...] = ()


BNA_HISTORY_URL = "https://www.bna.com.ar/Cotizador/DescargarPorFecha"


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


def _load_bna_divisa_buyer(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    """Load BNA's historical USD divisa buyer rate used by Rosario's Pizarra."""
    params = (
        ("id", "monedas"),
        ("fechaDesde", start_date.strftime("%d/%m/%Y")),
        ("fechaHasta", end_date.strftime("%d/%m/%Y")),
        ("idMonedaDescarga", "55"),  # Dolar U.S.A.
    )
    response = requests.get(
        BNA_HISTORY_URL,
        params=params,
        headers={"user-agent": "portfolio-analyzer/0.1"},
        timeout=30,
    )
    response.raise_for_status()

    bna = pd.read_csv(StringIO(response.text), sep=";", dtype=str)
    required_columns = {"Fecha cotizacion", "Compra"}
    if not required_columns.issubset(bna.columns):
        raise RuntimeError("BNA historical FX source returned an unexpected format.")

    bna["Fecha cotizacion"] = pd.to_datetime(
        bna["Fecha cotizacion"], dayfirst=True, errors="coerce"
    )
    bna["Compra"] = pd.to_numeric(
        bna["Compra"]
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )
    buyer_rate = (
        bna.dropna(subset=["Fecha cotizacion", "Compra"])
        .drop_duplicates(subset="Fecha cotizacion", keep="last")
        .set_index("Fecha cotizacion")["Compra"]
        .sort_index()
    )
    buyer_rate.name = "FX_ARS_USD"
    if buyer_rate.empty:
        raise RuntimeError("BNA historical FX source returned no USD buyer rates.")
    return buyer_rate


def _historical_levels(spread: pd.Series) -> tuple[float, float, float]:
    return (
        float(spread.quantile(0.25)),
        float(spread.median()),
        float(spread.quantile(0.75)),
    )


def _decision_from_percentile(percentile: float) -> str:
    if percentile <= 25:
        return "Venta favorable"
    if percentile >= 75:
        return "Esperar / precio local débil"
    return "Zona intermedia"


def fetch_soybean_spread(
    start_date: str = "2024-01-01",
    cache_file: str = "cache/chicago_data.csv",
) -> CommoditiesResult:
    start_dt = pd.Timestamp(start_date)
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
            .str.replace("Est.", "", regex=False)
            .str.strip()
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )
        .dropna()
        .sort_index()
        / 100
    )
    rosario_ars = rosario_ars[rosario_ars.index >= start_dt]
    if rosario_ars.empty:
        raise RuntimeError("No Rosario soybean data for selected period.")

    bna_buyer = _load_bna_divisa_buyer(
        start_date=rosario_ars.index.min(),
        end_date=rosario_ars.index.max(),
    )
    bna_aligned = bna_buyer.reindex(rosario_ars.index)
    rosario_usd = (rosario_ars / bna_aligned).dropna()
    rosario_ars = rosario_ars.reindex(rosario_usd.index)
    bna_aligned = bna_aligned.reindex(rosario_usd.index)
    rosario_usd, outlier_dates = _clean_isolated_price_outliers(rosario_usd)
    if outlier_dates:
        rosario_ars = rosario_ars.copy()
        rosario_ars.loc[list(outlier_dates)] = float("nan")
        rosario_ars = rosario_ars.interpolate(method="time", limit_area="inside")

    chicago_daily = chicago_usd_ton[chicago_usd_ton.index >= start_dt].squeeze()
    rosario_daily = rosario_usd[rosario_usd.index >= start_dt].squeeze()

    combined = pd.DataFrame(
        {
            "Chicago_USD_ton": chicago_daily,
            "Rosario_USD_ton": rosario_daily,
            "Rosario_ARS_ton": rosario_ars,
            "FX_ARS_USD": bna_aligned,
        }
    )
    combined.dropna(inplace=True)

    if combined.empty:
        raise RuntimeError("No aligned soybean data for selected period.")

    combined["Spread_USD"] = combined["Chicago_USD_ton"] - combined["Rosario_USD_ton"]
    combined["Spread_Pct"] = (combined["Spread_USD"] / combined["Chicago_USD_ton"]) * 100

    correlation = float(combined["Chicago_USD_ton"].corr(combined["Rosario_USD_ton"]))
    spread_latest = float(combined["Spread_Pct"].iloc[-1])
    spread_mean = float(combined["Spread_Pct"].mean())
    spread_percentile = float((combined["Spread_Pct"] <= spread_latest).mean() * 100)

    return CommoditiesResult(
        dataframe=combined,
        correlation=correlation,
        spread_pct_latest=spread_latest,
        spread_pct_mean=spread_mean,
        spread_percentile=spread_percentile,
        decision=_decision_from_percentile(spread_percentile),
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

    ax.set_title(
        "Precios diarios de soja: Chicago vs. Rosario",
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax.text(
        0.01,
        0.95,
        f"Brecha: USD {latest_spread:+.2f} ({latest_spread_pct:+.2f}%)\n"
        f"Rosario: ARS {latest_rosario_ars:,.0f}/tn\n"
        f"BNA divisas comprador: ARS {latest_fx:,.2f}/USD",
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
    lower_quartile, historic_median, upper_quartile = _historical_levels(spread)
    last_date = df.index[-1]
    last_spread = float(spread.iloc[-1])
    last_ema = float(ema20.iloc[-1])
    last_percentile = float((spread <= last_spread).mean() * 100)
    decision = _decision_from_percentile(last_percentile)
    latest_fx = float(df["FX_ARS_USD"].iloc[-1])

    lower_bound = min(0.0, float(spread.min()) - 1)
    upper_bound = max(40.0, float(spread.max()) + 2)
    ax.axhspan(
        lower_bound,
        lower_quartile,
        color="#27ae60",
        alpha=0.08,
        label=f"Venta favorable (P25: {lower_quartile:.1f}%)",
    )
    ax.axhspan(
        upper_quartile,
        upper_bound,
        color="#c0392b",
        alpha=0.08,
        label=f"Precio local débil (P75: {upper_quartile:.1f}%)",
    )
    ax.plot(df.index, spread, color="#2c3e50", linewidth=1.2, alpha=0.7, label="Descuento diario")
    ax.plot(df.index, ema20, color="#e74c3c", linewidth=2.2, label="EMA 20 dias")
    ax.axhline(
        historic_median,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"Mediana histórica: {historic_median:+.2f}%",
    )
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

    ax.set_title(
        "Descuento histórico de Rosario frente a Chicago",
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=20,
    )
    ax.text(
        0.01,
        0.95,
        f"Última rueda comparable: {last_date:%d %b %Y}\n"
        f"Descuento: {last_spread:+.2f}% | EMA20: {last_ema:+.2f}%\n"
        f"Posición histórica: P{last_percentile:.0f} | {decision}\n"
        f"BNA compra: ARS {latest_fx:,.2f}/USD",
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "#f8f9fa", "alpha": 0.95},
    )
    ax.set_ylabel("Descuento de Rosario (%)")
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
        fig.text(
            0.08,
            0.88,
            "Informe de soja: Chicago vs. Rosario",
            fontsize=24,
            fontweight="bold",
        )
        fig.text(
            0.08,
            0.83,
            f"Periodo analizado: {start_date} a {df.index[-1]:%Y-%m-%d}",
            fontsize=11,
            color="#555555",
        )
        summary = (
            f"Chicago: USD {float(latest['Chicago_USD_ton']):,.2f} por ton\n"
            f"Rosario: USD {float(latest['Rosario_USD_ton']):,.2f} por ton\n"
            f"Rosario: ARS {float(latest['Rosario_ARS_ton']):,.0f} por ton\n"
            f"BNA divisas comprador: ARS {float(latest['FX_ARS_USD']):,.2f} por USD\n"
            f"Descuento actual: {result.spread_pct_latest:+.2f}%\n"
            f"Posición histórica: P{result.spread_percentile:.0f} — {result.decision}"
        )
        fig.text(0.08, 0.58, summary, fontsize=14, linespacing=1.7, va="top")
        fig.text(
            0.08,
            0.16,
            "Fuentes: futuros de soja CBOT (Yahoo Finance), pizarra Rosario "
            "(Consiagro) y BNA divisas comprador.",
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
