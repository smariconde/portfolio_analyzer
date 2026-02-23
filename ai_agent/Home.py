import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ai_agent.env_loader import load_project_env

load_project_env()

st.set_page_config(page_title="AI Financial Agents", page_icon=":material/home:", layout="wide")


def render_home() -> None:
    st.markdown("# Financial Agents :material/finance_mode: :material/network_intelligence:")
    st.markdown(
        """
        #### AI Powered Finance Agents
        - Portfolio Agent (Buy / Hold / Sell)
        - Portfolio Optimizer
        - Valuation (DCF and multiples)
        - SEC Filings Analyzer
        - Sortino sector scanner
        - Commodities spread monitor (Chicago vs Rosario)
        - BCRA macro series dashboard
        - INDEC trade balance dashboard
        - Macro Dashboard Argentina (8 charts)
        """
    )


pages_by_group = {
    "Inicio": [
        st.Page(render_home, title="Overview", icon=":material/home:"),
    ],
    "Portfolio": [
        st.Page(
            "pages/01_portfolio_agent.py",
            title="Portfolio Agent",
            icon=":material/finance_mode:",
        ),
        st.Page(
            "pages/02_portfolio_optimizer.py",
            title="Portfolio Optimizer",
            icon=":material/functions:",
        ),
    ],
    "Valuation": [
        st.Page(
            "pages/03_valuation_agent.py",
            title="Valuation Agent",
            icon=":material/calculate:",
        ),
        st.Page(
            "pages/04_sec_filings.py",
            title="SEC Filings",
            icon=":material/history_edu:",
        ),
    ],
    "Market Research": [
        st.Page(
            "pages/10_sortino_scanner.py",
            title="Sortino Scanner",
            icon=":material/monitoring:",
        ),
        st.Page(
            "pages/11_commodities_spread.py",
            title="Commodities Spread",
            icon=":material/show_chart:",
        ),
    ],
    "Macro Argentina": [
        st.Page(
            "pages/12_bcra_macro.py",
            title="BCRA Macro",
            icon=":material/query_stats:",
        ),
        st.Page(
            "pages/13_indec_trade_balance.py",
            title="INDEC Trade Balance",
            icon=":material/public:",
        ),
        st.Page(
            "pages/14_macro_dashboard_argentina.py",
            title="Macro Dashboard AR",
            icon=":material/dashboard:",
        ),
    ],
}

fallback_links = {
    "Portfolio": [
        ("pages/01_portfolio_agent.py", "Portfolio Agent", ":material/finance_mode:"),
        ("pages/02_portfolio_optimizer.py", "Portfolio Optimizer", ":material/functions:"),
    ],
    "Valuation": [
        ("pages/03_valuation_agent.py", "Valuation Agent", ":material/calculate:"),
        ("pages/04_sec_filings.py", "SEC Filings", ":material/history_edu:"),
    ],
    "Market Research": [
        ("pages/10_sortino_scanner.py", "Sortino Scanner", ":material/monitoring:"),
        ("pages/11_commodities_spread.py", "Commodities Spread", ":material/show_chart:"),
    ],
    "Macro Argentina": [
        ("pages/12_bcra_macro.py", "BCRA Macro", ":material/query_stats:"),
        ("pages/13_indec_trade_balance.py", "INDEC Trade Balance", ":material/public:"),
        ("pages/14_macro_dashboard_argentina.py", "Macro Dashboard AR", ":material/dashboard:"),
    ],
}

with st.sidebar:
    st.divider()
    st.caption("Environment status")
    for key in ["GOOGLE_API_KEY", "TAVILY_API_KEY", "FINANCIAL_DATASETS_API_KEY"]:
        is_set = bool(os.environ.get(key))
        st.write(f"{'OK' if is_set else 'Missing'} `{key}`")
    model_hint = os.environ.get("GOOGLE_GENAI_MODEL") or os.environ.get("GEMINI_MODEL")
    if model_hint:
        st.caption(f"LLM model override: `{model_hint}`")

navigation = st.navigation(pages_by_group, position="sidebar")
runner = getattr(navigation, "run", None)

if callable(runner):
    runner()
elif isinstance(navigation, str):
    st.switch_page(navigation)
else:
    with st.sidebar:
        st.divider()
        st.caption("Navegacion (fallback)")
        for group, links in fallback_links.items():
            st.caption(group)
            for path, title, icon in links:
                st.page_link(path, label=title, icon=icon)
    render_home()
