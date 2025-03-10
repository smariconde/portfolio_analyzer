import os
import json
import re
from datetime import datetime
from typing import Dict, Any
import streamlit as st

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import get_financial_metrics, parse_output_to_json

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    max_tokens=None,
    max_retries=6,
    stop=None
)

def valuation_agent(ticker: str):
    """Calculates the intrinsic value of a company using a 5-year DCF model."""

    metrics = get_financial_metrics(ticker=ticker)

    # 1. Get necessary data from metrics
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

    # Get year
    current_year = datetime.now().year
    target_year = current_year + 5

    # 2. Estimate/Project Missing Metrics and Choose Valuation Method
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
                    "discount_rate": <float from 0.1 to 0.15> (Depending on the risk. For example, companies from less trusty countries, higher rate),
                    "terminal_growth_rate": <float>,
                    "valuation_method": <"PE_multiple" | "Gordon_Growth"> (Choose the best valuation method
                    based on the company. Use "PE_multiple" for mature companies with stable profitability
                     and "Gordon_Growth" for high-growth companies where profits are expected to grow into the future,
                    and do not use "PE_multiple" if there is no data for P/E)
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

                   Only include the revenue_cagr, profit_margin_{target_year}, shares_reduction_per_year, discount_rate, terminal_growth_rate and valuation_method in your JSON output.
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

    result = llm.invoke(prompt)
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


    # Project revenues and profits for next 5 years
    projected_revenues = []
    projected_profits = []

    for year in range(1, 6):
        if year == 1:
            revenue = ltm_revenue * (1 + revenue_cagr)
        else:
            revenue = projected_revenues[-1] * (1 + revenue_cagr)
        projected_revenues.append(revenue)

        profit = revenue * profit_margin_target_year
        projected_profits.append(profit)

    # Calculate the terminal value in the 5th year
    if valuation_method == "Gordon_Growth":
        terminal_value = (projected_profits[-1] * (1 + terminal_growth_rate)) / (
            discount_rate - terminal_growth_rate
        )
    elif valuation_method == "PE_multiple":
        terminal_value = projected_profits[-1] * pe_ratio_target_year
    else:
        terminal_value = 0  #Default value if method does not match

    # Calculate the present values for all the projected years and the terminal value
    present_values = []
    for year, profit in enumerate(projected_profits, 1):
        discounted_value = profit / (1 + discount_rate) ** year
        present_values.append(discounted_value)

    discounted_terminal_value = terminal_value / (1 + discount_rate) ** 5

    # Calculate intrinsic value
    intrinsic_value = sum(present_values) + discounted_terminal_value

    # Calculate intrinsic value per share considering reduction of shares
    shares_value = shares_outstanding * ((1 - shares_reduction_per_year) ** 5)

    intrinsic_value_per_share = intrinsic_value / shares_value

    # Get recomendation mean
    targetMeanPrice = metrics.get("targetMeanPrice", "Not Found")

    # Compare the intrinsic value with the current price
    undervalued = intrinsic_value_per_share > current_price

    message_content = {
        "intrinsic_value_per_share": round(intrinsic_value_per_share, 2),
        "current_price": current_price,
        "targetMeanPrice": targetMeanPrice,
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


if __name__ == "__main__":
    st.set_page_config(page_title="Valuation AI Agent", page_icon=":material/calculate:")
    st.title("Valuation Agent :material/calculate:")
    ticker = st.text_input("Enter Ticker", "AAPL", max_chars=5).upper()

    if st.button("Get Valuation", type="primary"):
        with st.spinner("Calculating Valuation..."):
            try:
                message_content, logo = valuation_agent(ticker=ticker)
                
                st.divider()
                col1, col2 = st.columns([1, 9], gap="small", vertical_alignment="center")
                with col1:
                     st.image(logo, width=100)
                with col2:
                    st.markdown(f"**[{ticker}](https://finviz.com/quote.ashx?t={ticker}&p=d)**")
                
                # Display metrics with conditional formatting
                col1, col2, col3 = st.columns(3)

                with col1:
                    if message_content["undervalued"]:
                         st.metric(label="Intrinsic Value", value=f"${message_content['intrinsic_value_per_share']:.2f}", delta="Undervalued", delta_color="normal")
                    else:
                         st.metric(label="Intrinsic Value", value=f"${message_content['intrinsic_value_per_share']:.2f}", delta="Overvalued", delta_color="inverse")

                with col2:
                    st.metric(label="Current Price", value=f"${message_content['current_price']:.2f}")

                with col3:
                    st.metric(label="Analyst Target", value=f"${message_content['targetMeanPrice']:.2f}")

                st.json(message_content)

            except Exception as e:
                st.error(f"An error occurred: {e}")