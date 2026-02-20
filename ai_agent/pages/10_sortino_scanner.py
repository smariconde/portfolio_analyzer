from __future__ import annotations

from pathlib import Path
import sys
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from portfolio_analyzer.analytics.sortino import DEFAULT_SECTORS, run_sortino_analysis


def _rank_global(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy().reset_index().rename(columns={"index": "Ticker"})
    ranked["2 Years"] = pd.to_numeric(ranked["2 Years"], errors="coerce")
    ranked["5 Years"] = pd.to_numeric(ranked["5 Years"], errors="coerce")
    ranked["Last Price"] = pd.to_numeric(ranked["Last Price"], errors="coerce")
    ranked["score_min"] = ranked[["2 Years", "5 Years"]].min(axis=1)
    ranked["score_avg"] = ranked[["2 Years", "5 Years"]].mean(axis=1)
    ranked = ranked.sort_values(
        by=["score_min", "score_avg", "5 Years", "2 Years"],
        ascending=False,
        na_position="last",
    )
    return ranked


def _finviz_chart_url(ticker: str) -> str:
    return (
        "https://charts2-node.finviz.com/chart.ashx"
        f"?cs=l&t={quote_plus(ticker)}&tf=d&s=linear&ct=candle_stick&tm=l"
        "&o[0][ot]=sma&o[0][op]=50&o[0][oc]=FF8F33C6"
        "&o[1][ot]=sma&o[1][op]=200&o[1][oc]=DCB3326D"
    )


def _read_legacy_cedears(path: Path) -> list[str]:
    if not path.exists():
        return []
    first_line = path.read_text(encoding="utf-8").splitlines()
    if not first_line:
        return []
    return [item.strip() for item in first_line[0].split(",") if item.strip()]


if "sortino_state" not in st.session_state:
    st.session_state.sortino_state = None

st.title("Sortino Scanner")
st.caption("Global Sortino scan with CEDEAR shortlist and visual review.")
st.info(
    "Now optimized for daily use: persistent results, global ranking, CEDEAR-focused output "
    "and a PDF report centered on shortlist review.",
    icon=":material/info:",
)
action_col1, action_col2 = st.columns([1, 1])
with action_col1:
    if st.button("Clear current analysis", icon=":material/delete_sweep:"):
        st.session_state.sortino_state = None
        st.rerun()
with action_col2:
    st.caption("The last successful run is kept until you clear it or run again.")

with st.form("sortino_form"):
    left, right = st.columns(2)
    with left:
        selected_sectors = st.multiselect("Sectors", DEFAULT_SECTORS, default=DEFAULT_SECTORS)
        generate_pdf = st.checkbox("Generate CEDEAR PDF report", value=True)
    with right:
        top_global = st.slider("Top rows (global ranking)", min_value=15, max_value=150, value=40, step=5)
        top_charts = st.slider("CEDEAR charts to display", min_value=3, max_value=24, value=9, step=3)

    submit = st.form_submit_button("Run Sortino Scan", type="primary")

if submit:
    if not selected_sectors:
        st.error("Select at least one sector.")
    else:
        with st.spinner("Running Sortino analysis..."):
            try:
                result = run_sortino_analysis(sectors=selected_sectors, generate_pdf=generate_pdf)
            except Exception as exc:
                st.error(f"Sortino analysis failed: {exc}")
                st.stop()

        df_result = getattr(result, "combined_df", None)
        if df_result is None:
            df_result = pd.read_csv(result.output_csv, index_col=0)

        cedears_result = getattr(result, "cedears_selected", None)
        if cedears_result is None:
            cedears_result = _read_legacy_cedears(result.output_txt)

        st.session_state.sortino_state = {
            "result": result,
            "df": df_result.copy(),
            "cedears": list(cedears_result),
            "top_global": top_global,
            "top_charts": top_charts,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        }
        st.success("Analysis completed and saved in session.")

state = st.session_state.sortino_state
if not state:
    st.warning("Run the scan to see ranking, CEDEAR charts and downloads.")
else:
    result = state["result"]
    df = state["df"].copy()
    cedears = sorted(set(state["cedears"]))
    ranked_global = _rank_global(df)
    ranked_cedears = ranked_global[ranked_global["Ticker"].isin(cedears)].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Tickers analyzed", int(result.total_tickers))
    col2.metric("Sectors with data", int(result.total_sectors))
    col3.metric("CEDEARs selected", int(len(cedears)))
    st.caption(f"Last run: `{state.get('generated_at', 'N/A')}`")

    if cedears:
        st.markdown(
            "Finviz screener: "
            f"[Open shortlist](https://finviz.com/screener.ashx?v=340&t={','.join(cedears)})"
        )

    tab1, tab2, tab3 = st.tabs(["CEDEAR Visual Board", "Global Ranking", "Downloads"])

    with tab1:
        if ranked_cedears.empty:
            st.warning("No CEDEARs met the current rule in this run.")
        else:
            st.subheader("Selected CEDEARs (ranked globally)")
            st.dataframe(
                ranked_cedears[
                    ["Ticker", "Sector", "2 Years", "5 Years", "score_min", "Last Price", "Last Date"]
                ].head(state["top_global"]),
                use_container_width=True,
                hide_index=True,
            )

            show_n = min(state["top_charts"], len(ranked_cedears))
            st.subheader(f"Top {show_n} CEDEAR charts")
            cols = st.columns(3)
            for idx, (_, row) in enumerate(ranked_cedears.head(show_n).iterrows()):
                ticker = str(row["Ticker"])
                with cols[idx % 3]:
                    st.markdown(f"**{ticker}**")
                    st.caption(
                        f"{row['Sector']} | 2Y: {row['2 Years']:.2f} | 5Y: {row['5 Years']:.2f}"
                    )
                    st.image(_finviz_chart_url(ticker), use_container_width=True)
                    st.markdown(f"[Finviz quote](https://finviz.com/quote.ashx?t={ticker}&p=d)")

    with tab2:
        st.subheader("Global ranking across all sectors")
        st.caption("Sorted by min(2Y, 5Y) Sortino. This avoids one-sector bias from raw CSV order.")
        st.dataframe(
            ranked_global[
                ["Ticker", "Sector", "2 Years", "5 Years", "score_min", "score_avg", "Last Price", "Last Date"]
            ].head(state["top_global"]),
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        st.subheader("Outputs")
        st.caption("Downloads do not clear the analysis anymore; results stay in the current session.")
        if result.output_csv.exists():
            st.download_button(
                "Download full Sortino CSV",
                data=result.output_csv.read_bytes(),
                file_name=result.output_csv.name,
                mime="text/csv",
            )
        if result.output_txt.exists():
            st.download_button(
                "Download CEDEAR shortlist TXT",
                data=result.output_txt.read_text(encoding="utf-8"),
                file_name=result.output_txt.name,
                mime="text/plain",
            )
        if result.output_pdf and result.output_pdf.exists():
            st.download_button(
                "Download CEDEAR PDF report",
                data=result.output_pdf.read_bytes(),
                file_name=result.output_pdf.name,
                mime="application/pdf",
            )
