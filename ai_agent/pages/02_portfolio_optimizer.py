from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from portfolio_analyzer.analytics.optimizer import optimize_portfolio


def main() -> None:
    st.title("Optimal Portfolio Optimizer")
    st.caption("Generate a Sortino-oriented allocation for a list of tickers.")
    st.info(
        "Tip: use 5-15 liquid tickers. The optimizer reserves a cash fraction and allocates the "
        "rest by risk-adjusted performance.",
        icon=":material/info:",
    )

    with st.form("optimizer_form"):
        left, right = st.columns(2)
        with left:
            portfolio_size = st.number_input(
                "Portfolio Size (USD)",
                min_value=1000.0,
                value=40000.0,
                step=1000.0,
            )
        with right:
            cash_percentage = st.slider(
                "Cash Percentage",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Fraction of portfolio to keep uninvested.",
            )
        default = "BRK-B,GOOGL,MSFT,LRCX,LLY,MELI,VIST,NU,COST"
        tickers_raw = st.text_area("Tickers (comma separated)", value=default)
        submit = st.form_submit_button("Calculate Portfolio", type="primary")

    if not submit:
        return

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        st.error("Enter at least one ticker.")
        return

    if len(set(tickers)) != len(tickers):
        st.warning("Duplicated tickers detected. Duplicates will be removed.")
        tickers = list(dict.fromkeys(tickers))

    if len(tickers) < 2:
        st.error("Enter at least two tickers to run optimization.")
        return

    investable = portfolio_size * (1 - cash_percentage)
    st.write(
        f"Universe: `{len(tickers)}` tickers | Investable capital: `${investable:,.2f}` "
        f"({(1 - cash_percentage) * 100:.1f}%)"
    )

    with st.spinner("Calculating optimal portfolio..."):
        try:
            result = optimize_portfolio(
                tickers=tickers,
                portfolio_size=portfolio_size,
                cash_percentage=cash_percentage,
            )
        except Exception as exc:
            st.error(f"Optimizer failed: {exc}")
            return

    st.subheader("Optimal Portfolio Allocation")
    table = result.weights.copy()
    if "weight" in table.columns:
        table["weight_pct"] = pd.to_numeric(table["weight"], errors="coerce") * 100
    if "investment" in table.columns:
        table["investment"] = pd.to_numeric(table["investment"], errors="coerce")
    table = table.sort_values(by="investment", ascending=False, na_position="last")
    st.dataframe(table, use_container_width=True)

    total_invested = float(result.weights["investment"].sum())
    total_cash = portfolio_size - total_invested

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Invested", f"${total_invested:,.2f}")
    col2.metric("Total Cash", f"${total_cash:,.2f}")
    col3.metric("Portfolio Sortino", f"{result.sortino_ratio:.2f}")
    col4.metric("S&P 500 Sortino", f"{result.benchmark_sortino:.2f}")
    st.metric("Sortino Edge vs Benchmark", f"{(result.sortino_ratio - result.benchmark_sortino):.2f}")

    if "weight_pct" in table.columns:
        chart_df = table[["weight_pct"]].dropna()
        if not chart_df.empty:
            st.bar_chart(chart_df)

    csv = result.weights.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download allocation CSV",
        data=csv,
        file_name="portfolio_allocation.csv",
        mime="text/csv",
    )


if __name__ in {"__main__", "__page__"}:
    main()
