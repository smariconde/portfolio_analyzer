import os
import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ai_agent.env_loader import load_project_env
from ai_agent.llm_utils import invoke_gemini

load_project_env()
from langchain_core.prompts import ChatPromptTemplate

from ai_agent.tools import get_financial_metrics, parse_output_to_json

api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

def valuation_agent(ticker: str):
    """Calculates the intrinsic value of a company using a 5-year DCF model."""
    metrics = get_financial_metrics(ticker=ticker)

    ltm_revenue = metrics.get("totalRevenue") or 0
    shares_outstanding = metrics.get("sharesOutstanding") or 0
    current_price = metrics.get("currentPrice") or 0
    trailing_pe = metrics.get("price_to_earnings_ratio") or 0
    revenue_growth = metrics.get("revenue_growth") or 0
    earnings_growth = metrics.get("earnings_growth") or 0
    current_profit_margin = metrics.get("profitMargins") or 0
    industry = metrics.get("industry", "Not Found")
    sector = metrics.get("sector", "Not Found")
    logo = metrics.get("logo_url")

    current_year = datetime.now().year
    target_year = current_year + 5

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an expert financial analyst. Your job is to analyze a company's
                    metrics, maturity, size, industry consensus, and risk to estimate the
                    following for a 5-year DCF model:
                    "revenue_cagr": <float>,
                    "profit_margin_{target_year}": <float>,
                    "shares_reduction_per_year": <float>,
                    "discount_rate": <float from 0.1 to 0.15>,
                    "terminal_growth_rate": <float>,
                    "valuation_method": <"PE_multiple" | "Gordon_Growth">
                    Give me the JSON output only.
                """,
            ),
            (
                "human",
                f"""Based on the company below, provide the data for the valuation model.
                    Company: {ticker}
                    Industry: {industry}
                    Sector: {sector}

                    LTM Revenue: {ltm_revenue}
                    Shares Outstanding: {shares_outstanding}
                    Trailing P/E Ratio: {trailing_pe}
                    Revenue Growth: {revenue_growth}
                    Earnings Growth: {earnings_growth}
                    Current Profit Margin: {current_profit_margin}

                   Only include the required keys in JSON output.
                   Do not include any JSON markdown.
                """,
            ),
        ]
    )

    prompt = template.invoke(
        {
            "ticker": ticker,
            "industry": industry,
            "sector": sector,
            "ltm_revenue": ltm_revenue,
            "shares_outstanding": shares_outstanding,
            "trailing_pe": trailing_pe,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "current_profit_margin": current_profit_margin,
            "target_year": target_year,
        }
    )

    result, _ = invoke_gemini(prompt, temperature=0, max_tokens=None, max_retries=6, stop=None)
    try:
        estimation = parse_output_to_json(result.content)
    except json.JSONDecodeError as e:
        estimation = {"error": f"JSON Decode Error {e}", "result": result.content}

    revenue_cagr = estimation.get("revenue_cagr", 0.05)
    profit_margin_target_year = estimation.get(f"profit_margin_{target_year}", 0.15)
    shares_reduction_per_year = estimation.get("shares_reduction_per_year", 0.01)
    discount_rate = estimation.get("discount_rate", 0.1)
    terminal_growth_rate = estimation.get("terminal_growth_rate", 0.03)
    valuation_method = estimation.get("valuation_method", "PE_multiple")
    pe_ratio_target_year = estimation.get(f"pe_ratio_{target_year}", 15)

    projected_revenues = []
    projected_profits = []

    for year in range(1, 6):
        if year == 1:
            revenue = ltm_revenue * (1 + revenue_cagr)
        else:
            revenue = projected_revenues[-1] * (1 + revenue_cagr)
        projected_revenues.append(revenue)
        projected_profits.append(revenue * profit_margin_target_year)

    if valuation_method == "Gordon_Growth":
        terminal_value = (projected_profits[-1] * (1 + terminal_growth_rate)) / (
            discount_rate - terminal_growth_rate
        )
    elif valuation_method == "PE_multiple":
        terminal_value = projected_profits[-1] * pe_ratio_target_year
    else:
        terminal_value = 0

    present_values = []
    for year, profit in enumerate(projected_profits, 1):
        present_values.append(profit / (1 + discount_rate) ** year)

    discounted_terminal_value = terminal_value / (1 + discount_rate) ** 5
    intrinsic_value = sum(present_values) + discounted_terminal_value

    shares_value = shares_outstanding * ((1 - shares_reduction_per_year) ** 5)
    intrinsic_value_per_share = intrinsic_value / shares_value if shares_value else 0

    target_mean_price = metrics.get("targetMeanPrice", None)
    undervalued = intrinsic_value_per_share > current_price if current_price else False

    message_content = {
        "intrinsic_value_per_share": round(intrinsic_value_per_share, 2),
        "current_price": current_price,
        "targetMeanPrice": target_mean_price,
        "undervalued": undervalued,
        "valuation_method": valuation_method,
        "reasoning": {
            "estimated_metrics": estimation,
            "ltm_revenue": ltm_revenue,
            "shares_outstanding": shares_outstanding,
            "trailing_pe": trailing_pe,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "current_profit_margin": current_profit_margin,
        },
    }

    return message_content, logo


@st.cache_data(ttl=21600)
def load_valuation_cached(ticker: str):
    return valuation_agent(ticker)


if __name__ in {"__main__", "__page__"}:
    st.title("Valuation Agent :material/calculate:")
    st.caption("Estimate intrinsic value using a 5-year AI-assisted DCF workflow.")
    st.info(
        "Output is a scenario-based estimate, not investment advice. Validate assumptions before use.",
        icon=":material/info:",
    )

    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning("`GOOGLE_API_KEY` is missing. Valuation requests will fail.")
    if st.button("Refresh valuation cache", icon=":material/refresh:"):
        st.cache_data.clear()
        st.rerun()

    with st.form("valuation_form"):
        ticker = st.text_input("Ticker", "AAPL", max_chars=8).upper().strip()
        submit = st.form_submit_button("Get Valuation", type="primary")

    if submit:
        if not ticker:
            st.error("Enter a valid ticker.")
        else:
            with st.spinner("Calculating valuation..."):
                try:
                    message_content, logo = load_valuation_cached(ticker=ticker)
                except Exception as exc:
                    st.error(f"Valuation failed: {exc}")
                    st.stop()

            st.divider()
            col1, col2 = st.columns([1, 9], gap="small", vertical_alignment="center")
            with col1:
                if logo:
                    st.image(logo, width=100)
            with col2:
                st.markdown(f"**[{ticker}](https://finviz.com/quote.ashx?t={ticker}&p=d)**")

            c1, c2, c3 = st.columns(3)
            with c1:
                delta_text = "Undervalued" if message_content["undervalued"] else "Overvalued"
                delta_color = "normal" if message_content["undervalued"] else "inverse"
                st.metric(
                    label="Intrinsic Value",
                    value=f"${message_content['intrinsic_value_per_share']:.2f}",
                    delta=delta_text,
                    delta_color=delta_color,
                )
            with c2:
                current_price = message_content.get("current_price")
                value = f"${float(current_price):.2f}" if isinstance(current_price, (int, float)) else "N/A"
                st.metric(label="Current Price", value=value)
            with c3:
                target = message_content.get("targetMeanPrice")
                value = f"${float(target):.2f}" if isinstance(target, (int, float)) else "N/A"
                st.metric(label="Analyst Target", value=value)

            current_price = message_content.get("current_price")
            intrinsic = float(message_content["intrinsic_value_per_share"])
            if isinstance(current_price, (int, float)) and current_price:
                upside = ((intrinsic / float(current_price)) - 1) * 100
                st.metric("Implied Upside vs Current", f"{upside:.2f}%")

            tab1, tab2 = st.tabs(["Summary", "Raw JSON"])
            with tab1:
                st.write(
                    {
                        "valuation_method": message_content["valuation_method"],
                        "undervalued": message_content["undervalued"],
                        "intrinsic_value_per_share": message_content["intrinsic_value_per_share"],
                        "current_price": message_content.get("current_price"),
                        "targetMeanPrice": message_content.get("targetMeanPrice"),
                    }
                )
            with tab2:
                st.json(message_content)
