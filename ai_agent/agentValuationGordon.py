import os, json, re
from datetime import datetime
from typing import Dict, Any
import streamlit as st

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import get_financial_metrics

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-002",
    temperature=0,
    max_tokens=None,
    max_retries=6,
    stop=None
)
    
def parse_output_to_json(output):
    # Si ya es un diccionario o lista, devolverlo tal cual
    if isinstance(output, (dict, list)):
        return output
    
    # Si es una cadena, intentar parsearla de múltiples formas
    if isinstance(output, str):
        # Limpiar la cadena de posibles marcadores de código
        cleaned_output = re.sub(r"```(json)?", '', output).strip()
        
        try:
            # Intentar parsear directamente
            return json.loads(cleaned_output)
        except json.JSONDecodeError:
            # Si falla, intentar extraer un JSON entre llaves
            try:
                match = re.search(r'(\{.*\})', cleaned_output, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
            except:
                pass
    
    # Si no se puede convertir a JSON, convertir a diccionario simple
    return {"raw_output": str(output)}


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
    
    # 2. Estimate/Project Missing Metrics

    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                 f"""You are an expert financial analyst.
                Your job is to take a look at the company metrics, maturity, size, industry consensus and risk to
                estimate the following:
                "revenue_cagr": <float>,
                "profit_margin_{target_year}": <float>,
                "shares_reduction_per_year": <float>,
                "discount_rate": <float> (Depending on the risk),
                "terminal_growth_rate": <float> (Depending on the maturity of the company)
                Give me the JSON output only.
                """
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

    
    
                   Only include the revenue_cagr, profit_margin_{target_year}, shares_reduction_per_year, discount_rate and terminal_growth_rate in your JSON output.
                   Do not include any JSON markdown.
                """
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "ticker": ticker,
            "industry": metrics.get("industry", "Not Found"),
            "sector": metrics.get("sector", "Not Found"),
            "ltm_revenue": ltm_revenue,
            "shares_outstanding": shares_outstanding,
            "trailing_pe": trailing_pe,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "current_profit_margin": current_profit_margin,
            "target_year": target_year
        }
    )

    # Invoke the LLM
    result = llm.invoke(prompt)
    try:
        estimation = parse_output_to_json(result.content)
    except json.JSONDecodeError as e:
        estimation = {"error": f"JSON Decode Error {e}", "result": result.content}
    
    revenue_cagr = estimation.get("revenue_cagr", 0.05)
    profit_margin_target_year = estimation.get(f"profit_margin_{target_year}", 0.15)
    shares_reduction_per_year = estimation.get("shares_reduction_per_year", 0.01) # Assuming a share reduction of 1%
    discount_rate = estimation.get("discount_rate", 0.1)
    terminal_growth_rate = estimation.get("terminal_growth_rate", 0.03)
    
    # Project revenues and profits for next 5 years
    projected_revenues = []
    projected_profits = []
    
    # Calculate for the next 5 years
    for year in range(1, 6):
        if year == 1:
            revenue = ltm_revenue * (1 + revenue_cagr)
        else:
            revenue = projected_revenues[-1] * (1 + revenue_cagr)
        projected_revenues.append(revenue)

        profit = revenue * profit_margin_target_year
        projected_profits.append(profit)

    # Calculate the terminal value in the 5th year
    # terminal_value = projected_profits[-1] * pe_ratio_target_year # Removed
    terminal_value = (projected_profits[-1] * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    
    #Calculate the present values for all the projected years and the terminal value
    present_values = []
    for year, profit in enumerate(projected_profits,1):
        discounted_value = profit / (1 + discount_rate) ** year
        present_values.append(discounted_value)
        
    discounted_terminal_value = terminal_value / (1 + discount_rate) ** 5
    
    # Calculate intrinsic value
    intrinsic_value = sum(present_values) + discounted_terminal_value
    
    # Calculate intrinsic value per share considering reduction of shares
    shares_value = shares_outstanding * ((1 - shares_reduction_per_year) ** 5)
    
    intrinsic_value_per_share = intrinsic_value/shares_value

    # Get recomendation mean
    targetMeanPrice = metrics.get("targetMeanPrice", "Not Found")

    # Compare the intrinsic value with the current price
    undervalued = intrinsic_value_per_share > current_price
    
    message_content = {
        "intrinsic_value_per_share": round(intrinsic_value_per_share,2),
        "current_price": current_price,
        "targetMeanPrice": targetMeanPrice,
        "undervalued": undervalued,
        "reasoning": {
            "estimated_metrics": estimation,
             "ltm_revenue": ltm_revenue,
             "shares_outstanding": shares_outstanding
        }
    }


    return message_content, logo


if __name__ == "__main__":
    st.title("Valuation Agent")
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
               
                st.json(message_content)

            except Exception as e:
                st.error(f"An error occurred: {e}")