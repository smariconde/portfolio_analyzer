from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from portfolio_analyzer.data.commodities import fetch_soybean_spread


@st.cache_data(ttl=3600)
def load_spread(start_date_iso: str):
    return fetch_soybean_spread(start_date=start_date_iso)


st.title("Commodities Spread: Soybean")
st.caption("Daily Chicago vs Rosario soybean spread in USD/ton.")
st.info(
    "Track relative soybean pricing between Chicago and Rosario in USD to monitor dislocations.",
    icon=":material/info:",
)
if st.button("Refresh cached data", icon=":material/refresh:"):
    st.cache_data.clear()
    st.rerun()

with st.form("commodities_form"):
    left, right = st.columns(2)
    with left:
        start_date = st.date_input("Start date", value=date(2024, 1, 1))
    with right:
        rows = st.slider("Rows to preview", min_value=10, max_value=180, value=45, step=15)
    submit = st.form_submit_button("Run spread analysis", type="primary")

if submit:
    with st.spinner("Downloading and processing data..."):
        try:
            result = load_spread(start_date.strftime("%Y-%m-%d"))
        except Exception as exc:
            st.error(f"Commodities analysis failed: {exc}")
            st.stop()

    st.success("Analysis completed")
    latest_row = result.dataframe.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Correlation", f"{result.correlation:.4f}")
    col2.metric("Latest Spread %", f"{result.spread_pct_latest:.2f}%")
    col3.metric("Mean Spread %", f"{result.spread_pct_mean:.2f}%")
    c1, c2 = st.columns(2)
    c1.metric("Chicago latest (USD/ton)", f"{float(latest_row['Chicago_USD_ton']):,.2f}")
    c2.metric("Rosario latest (USD/ton)", f"{float(latest_row['Rosario_USD_ton']):,.2f}")

    st.subheader("Price series")
    st.line_chart(result.dataframe[["Chicago_USD_ton", "Rosario_USD_ton"]])
    st.subheader("Spread %")
    st.line_chart(result.dataframe[["Spread_Pct"]])
    st.dataframe(result.dataframe.tail(rows), use_container_width=True)

    csv_data = result.dataframe.to_csv(index=True).encode("utf-8")
    st.download_button(
        "Download spread data CSV",
        data=csv_data,
        file_name="soybean_spread.csv",
        mime="text/csv",
    )
