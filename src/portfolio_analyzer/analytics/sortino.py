from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import sortino as legacy_sortino

DEFAULT_SECTORS = [
    "Industrials",
    "Health Care",
    "Information Technology",
    "Utilities",
    "Financials",
    "Materials",
    "Consumer Discretionary",
    "Real Estate",
    "Communication Services",
    "Consumer Staples",
    "Energy",
]


@dataclass
class SortinoRunResult:
    output_csv: Path
    output_txt: Path
    output_pdf: Path | None
    total_tickers: int
    total_sectors: int
    combined_df: pd.DataFrame
    cedears_selected: list[str]


def _to_ranked_df(combined_df: pd.DataFrame, cedears: list[str]) -> pd.DataFrame:
    ranked = combined_df.copy()
    ranked["2 Years"] = pd.to_numeric(ranked["2 Years"], errors="coerce")
    ranked["5 Years"] = pd.to_numeric(ranked["5 Years"], errors="coerce")
    ranked["Last Price"] = pd.to_numeric(ranked["Last Price"], errors="coerce")
    ranked = ranked.replace([np.inf, -np.inf], np.nan)
    ranked["score_min"] = ranked[["2 Years", "5 Years"]].min(axis=1)
    ranked["score_avg"] = ranked[["2 Years", "5 Years"]].mean(axis=1)
    ranked = ranked.reset_index().rename(columns={"index": "Ticker"})
    ranked = ranked[ranked["Ticker"].isin(set(cedears))]
    ranked = ranked.sort_values(
        by=["score_min", "score_avg", "5 Years", "2 Years"],
        ascending=False,
        na_position="last",
    )
    return ranked


def _generate_cedears_pdf(
    combined_df: pd.DataFrame,
    cedears: list[str],
    analysis_date: str,
    output_pdf: Path,
) -> Path:
    ranked = _to_ranked_df(combined_df, cedears)

    class ReportPDF(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", size=8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, f"Page {self.page_no()}", align="C")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    # Cover
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Sortino Scanner - CEDEAR Report", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Analysis date: {analysis_date}", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, f"Tickers analyzed: {len(combined_df)}", ln=True)
    pdf.cell(0, 8, f"Sectors with data: {combined_df['Sector'].nunique()}", ln=True)
    pdf.cell(0, 8, f"CEDEARs selected: {len(cedears)}", ln=True)
    pdf.ln(4)
    pdf.multi_cell(
        0,
        7,
        "Selection rule: CEDEARs that beat S&P 500 in both 2Y and 5Y Sortino ratios.",
    )
    if cedears:
        screener_link = "https://finviz.com/screener.ashx?v=340&t=" + ",".join(sorted(set(cedears)))
        pdf.multi_cell(0, 7, f"Finviz screener link: {screener_link}")
    else:
        pdf.multi_cell(0, 7, "No CEDEARs met the rule for this run.")

    # Ranked CEDEAR table
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Ranked CEDEARs (Global)", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(24, 7, "Ticker", border=1)
    pdf.cell(42, 7, "Sector", border=1)
    pdf.cell(18, 7, "2Y", border=1, align="R")
    pdf.cell(18, 7, "5Y", border=1, align="R")
    pdf.cell(18, 7, "Score", border=1, align="R")
    pdf.cell(25, 7, "Price", border=1, align="R")
    pdf.cell(35, 7, "Date", border=1)
    pdf.ln(7)

    pdf.set_font("Helvetica", size=8)
    for _, row in ranked.head(60).iterrows():
        if pdf.get_y() > 265:
            pdf.add_page()
        pdf.cell(24, 6, str(row["Ticker"]), border=1)
        pdf.cell(42, 6, str(row.get("Sector", ""))[:22], border=1)
        pdf.cell(18, 6, f"{row.get('2 Years', float('nan')):.2f}", border=1, align="R")
        pdf.cell(18, 6, f"{row.get('5 Years', float('nan')):.2f}", border=1, align="R")
        pdf.cell(18, 6, f"{row.get('score_min', float('nan')):.2f}", border=1, align="R")
        price = row.get("Last Price", float("nan"))
        price_txt = f"{price:.2f}" if pd.notna(price) else "N/A"
        pdf.cell(25, 6, price_txt, border=1, align="R")
        pdf.cell(35, 6, str(row.get("Last Date", ""))[:10], border=1)
        pdf.ln(6)

    # Finviz chart pages
    for _, row in ranked.head(12).iterrows():
        ticker = str(row["Ticker"])
        chart_url = (
            "https://charts2-node.finviz.com/chart.ashx"
            f"?cs=l&t={quote_plus(ticker)}&tf=d&s=linear&ct=candle_stick&tm=l"
            "&o[0][ot]=sma&o[0][op]=50&o[0][oc]=FF8F33C6"
            "&o[1][ot]=sma&o[1][op]=200&o[1][oc]=DCB3326D"
        )
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, f"{ticker} - Finviz Chart", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(
            0,
            7,
            f"Sector: {row.get('Sector', '')} | 2Y: {row.get('2 Years', float('nan')):.2f} "
            f"| 5Y: {row.get('5 Years', float('nan')):.2f}",
            ln=True,
        )
        try:
            response = requests.get(chart_url, timeout=20)
            response.raise_for_status()
            suffix = ".png"
            with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            pdf.image(tmp_path, x=8, y=28, w=194)
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pdf.set_font("Helvetica", size=9)
            pdf.multi_cell(0, 6, f"Unable to fetch Finviz chart.\nURL: {chart_url}")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_pdf))
    return output_pdf


def run_sortino_analysis(
    sectors: list[str] | None = None,
    output_csv: str = "outputs/sortino/all_sortino_ratios.csv",
    output_txt: str = "outputs/sortino/cedears_selection.txt",
    generate_pdf: bool = True,
) -> SortinoRunResult:
    """Run Sortino analysis using the existing battle-tested legacy implementation."""
    selected_sectors = sectors or DEFAULT_SECTORS
    all_sector_dfs: list[pd.DataFrame] = []
    cedears_selection_global: list[str] = []

    today = datetime.today()
    two_years_ago = today - timedelta(days=2 * 365)
    five_years_ago = today - timedelta(days=5 * 365)

    sp500_data = legacy_sortino.download_ticker_with_cache(
        "^GSPC",
        start=five_years_ago.strftime("%Y-%m-%d"),
        end=today.strftime("%Y-%m-%d"),
    )
    if sp500_data is None or sp500_data.empty:
        raise RuntimeError("Failed to download S&P 500 data.")

    if isinstance(sp500_data, pd.Series):
        sp500_close = sp500_data
    elif "close" in sp500_data.columns:
        sp500_close = sp500_data["close"]
    elif "Close" in sp500_data.columns:
        sp500_close = sp500_data["Close"]
    elif "adjclose" in sp500_data.columns:
        sp500_close = sp500_data["adjclose"]
    else:
        sp500_close = sp500_data.iloc[:, 0]

    sp500_close.index = pd.DatetimeIndex(sp500_close.index)
    sp500_returns = sp500_close.pct_change()

    risk_free_rate = 0.02
    downside_sp500 = sp500_returns[sp500_returns < 0].std() * np.sqrt(252)
    two_years_ago_dt = pd.Timestamp(two_years_ago)

    sortino_sp500_2yr = (
        (sp500_returns[sp500_returns.index >= two_years_ago_dt].mean() * 252 - risk_free_rate)
        / downside_sp500
    )
    sortino_sp500_5yr = ((sp500_returns.mean() * 252 - risk_free_rate) / downside_sp500)

    for sector in selected_sectors:
        try:
            sector_df, sector_cedears = legacy_sortino.analyze_sector_with_sortino(
                sector,
                sp500_close,
                sp500_returns,
                sortino_sp500_2yr,
                sortino_sp500_5yr,
                two_years_ago_dt,
            )
        except Exception:
            continue

        if sector_df is None or sector_df.empty:
            continue

        sector_df["Sector"] = sector
        all_sector_dfs.append(sector_df)
        cedears_selection_global.extend(sector_cedears)

    if not all_sector_dfs:
        raise RuntimeError("Sortino analysis finished without usable sector data.")

    combined_sortino_df = pd.concat(all_sector_dfs)
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined_sortino_df.to_csv(output_csv_path)

    output_txt_path = Path(output_txt)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with output_txt_path.open("w", encoding="utf-8") as file:
        unique_cedears = sorted(list(set(cedears_selection_global)))
        if unique_cedears:
            file.write(", ".join(unique_cedears))
            file.write("\nhttps://finviz.com/screener.ashx?v=340&t=" + ",".join(unique_cedears))
        else:
            file.write("No CEDEARs met the selection criteria.")

    unique_cedears = sorted(list(set(cedears_selection_global)))

    output_pdf: Path | None = None
    if generate_pdf:
        analysis_date = datetime.today().strftime("%Y-%m-%d")
        pdf_filename = _generate_cedears_pdf(
            combined_df=combined_sortino_df,
            cedears=unique_cedears,
            analysis_date=analysis_date,
            output_pdf=Path(f"outputs/sortino/Sortino_CEDEAR_Report_{analysis_date}.pdf"),
        )
        output_pdf = pdf_filename

    return SortinoRunResult(
        output_csv=output_csv_path,
        output_txt=output_txt_path,
        output_pdf=output_pdf,
        total_tickers=len(combined_sortino_df),
        total_sectors=len(all_sector_dfs),
        combined_df=combined_sortino_df,
        cedears_selected=unique_cedears,
    )
