from typing import Annotated, Any, Dict, Sequence, TypedDict

import operator, os, json, re, time, math
import sys
from pathlib import Path
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ai_agent.tools import calculate_bollinger_bands, calculate_macd, calculate_obv, calculate_rsi, get_price_data, prices_to_df, get_financial_metrics, format_metric, get_news, calculate_trend_signals, calculate_mean_reversion_signals, calculate_momentum_signals, calculate_volatility_signals, calculate_stat_arb_signals, weighted_signal_combination, normalize_pandas, parse_output_to_json
from ai_agent.llm_utils import invoke_gemini
from ai_agent.ui_helpers import build_finviz_chart_url, extract_summary_fields
from ai_agent.valuation_v2.assistant import (
    deterministic_method_fallback,
    propose_method_and_assumptions_with_llm,
)
from ai_agent.valuation_v2.engine import run_valuation

import streamlit as st

from ai_agent.env_loader import load_project_env

load_project_env()
from datetime import datetime, timedelta



api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {**a, **b}

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]
    agent_reasoning: Annotated[Dict[str,Any], merge_dicts]

##### Market Data Agent #####
def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing market data"""
    messages = state["messages"]
    data = state["data"]

    # Set default dates
    end_date = data["end_date"] or datetime.now().strftime('%Y-%m-%d')
    if not data["start_date"]:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = end_date_obj.replace(month=end_date_obj.month - 3) if end_date_obj.month > 3 else \
            end_date_obj.replace(year=end_date_obj.year - 1, month=end_date_obj.month + 9)
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        start_date = data["start_date"]

    # Get the historical price data
    prices = get_price_data(data["ticker"], start_date, end_date)
    # Get the financial metrics
    financial_metrics = get_financial_metrics(
        ticker=data["ticker"]
    )
    # Get Market news of the stock
    market_news = get_news(
        query=f"Show me ${data['ticker']} financial, operational and related bussines news.",
        max_results=5,
    )

    return {
        "messages": messages,
        "data": {
            **data, 
            "prices": prices, 
            "start_date": start_date, 
            "end_date": end_date,
            "financial_metrics": financial_metrics,
            "market_news": market_news
        }
    }

##### Quantitative Agent #####
def quant_agent(state: AgentState):
    """Analyzes technical indicators and generates trading signals."""
    show_reasoning = state["metadata"]["show_reasoning"]

    data = state["data"]
    prices = data["prices"]
    prices_df = prices_to_df(prices)
    print(prices_df)
    current_price = prices_df['close'].iloc[-1].item()

    # Calculate indicators
    # 1. MACD (Moving Average Convergence Divergence)
    macd_line, signal_line = calculate_macd(prices_df)
    
    # 2. RSI (Relative Strength Index)
    rsi = calculate_rsi(prices_df)
    
    # 3. Bollinger Bands (Bollinger Bands)
    upper_band, lower_band = calculate_bollinger_bands(prices_df)
    
    # 4. OBV (On-Balance Volume)
    obv = calculate_obv(prices_df)
    
    # Generate individual signals
    signals = []
    
    # MACD signal
    if macd_line.iloc[-2].item() < signal_line.iloc[-2].item() and macd_line.iloc[-1].item() > signal_line.iloc[-1].item():
        signals.append('bullish')
    elif macd_line.iloc[-2].item() > signal_line.iloc[-2].item() and macd_line.iloc[-1].item() < signal_line.iloc[-1].item():
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # RSI signal
    if rsi.iloc[-1].item() < 30:
        signals.append('bullish')
    elif rsi.iloc[-1].item() > 70:
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # Bollinger Bands signal
    current_price = prices_df['close'].iloc[-1].item()
    if current_price < lower_band.iloc[-1].item():
        signals.append('bullish')
    elif current_price > upper_band.iloc[-1].item():
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # OBV signal
    obv_slope = obv.diff().iloc[-5:].mean()
    if obv_slope > 0:
        signals.append('bullish')
    elif obv_slope < 0:
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # Add reasoning collection
    reasoning = {
        "MACD": {
            "signal": signals[0],
            "details": f"MACD Line crossed {'above' if signals[0] == 'bullish' else 'below' if signals[0] == 'bearish' else 'neither above nor below'} Signal Line"
        },
        "RSI": {
            "signal": signals[1],
            "details": f"RSI is {rsi.iloc[-1].item():.2f} ({'oversold' if signals[1] == 'bullish' else 'overbought' if signals[1] == 'bearish' else 'neutral'})"
        },
        "Bollinger": {
            "signal": signals[2],
            "details": f"Price is {'below lower band' if signals[2] == 'bullish' else 'above upper band' if signals[2] == 'bearish' else 'within bands'}"
        },
        "OBV": {
            "signal": signals[3],
            "details": f"OBV slope is {obv_slope:.2f} ({signals[3]})"
        },
        "Current Price": current_price
    }
    
    # Determine overall signal
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    
    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    # Calculate confidence level based on the proportion of indicators agreeing
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals
    
    # Generate the message content
    message_content = {
        "signal": overall_signal,
        "confidence": round(confidence, 2),
        "reasoning": {
            "MACD": reasoning["MACD"],
            "RSI": reasoning["RSI"],
            "Bollinger": reasoning["Bollinger"],
            "OBV": reasoning["OBV"],
            "Current price": reasoning["Current Price"]
        }
    }

    # 1. Trend Following Strategy
    trend_signals = calculate_trend_signals(prices_df)

    # 2. Mean Reversion Strategy
    mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

    # 3. Momentum Strategy
    momentum_signals = calculate_momentum_signals(prices_df)

    # 4. Volatility Strategy
    volatility_signals = calculate_volatility_signals(prices_df)

    # 5. Statistical Arbitrage Signals
    stat_arb_signals = calculate_stat_arb_signals(prices_df)

    # Combine all signals using a weighted ensemble approach
    strategy_weights = {
        'trend': 0.25,
        'mean_reversion': 0.20,
        'momentum': 0.25,
        'volatility': 0.15,
        'stat_arb': 0.15
    }

    combined_signal = weighted_signal_combination({
        'trend': trend_signals,
        'mean_reversion': mean_reversion_signals,
        'momentum': momentum_signals,
        'volatility': volatility_signals,
        'stat_arb': stat_arb_signals
    }, strategy_weights)

    # Generate detailed analysis report
    analysis_report = {
        "signal": combined_signal['signal'],
        "confidence": f"{round(combined_signal['confidence'] * 100)}%",
        "strategy_signals": {
            "trend_following": {
                "signal": trend_signals['signal'],
                "confidence": f"{round(trend_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(trend_signals['metrics'])
            },
            "mean_reversion": {
                "signal": mean_reversion_signals['signal'],
                "confidence": f"{round(mean_reversion_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(mean_reversion_signals['metrics'])
            },
            "momentum": {
                "signal": momentum_signals['signal'],
                "confidence": f"{round(momentum_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(momentum_signals['metrics'])
            },
            "volatility": {
                "signal": volatility_signals['signal'],
                "confidence": f"{round(volatility_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(volatility_signals['metrics'])
            },
            "statistical_arbitrage": {
                "signal": stat_arb_signals['signal'],
                "confidence": f"{round(stat_arb_signals['confidence'] * 100)}%",
                "metrics": normalize_pandas(stat_arb_signals['metrics'])
            }
        },
        "current_price": round(current_price, 2),

    }

    # Create the quant message
    message = HumanMessage(
        content=str(analysis_report),  # Convert dict to string for message content
        name="quant_agent",
    )

    reasoning_output = show_agent_reasoning(analysis_report) if show_reasoning else None

    return {
        "messages": state["messages"] + [message],
        "data": data,
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "quant_agent": reasoning_output
            }
    }

##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals."""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"]  # Get the most recent metrics

    # Initialize signals list for different fundamental aspects
    signals = []
    reasoning = {}
    
    # 1. Profitability Analysis
    profitability_score = 0
    if metrics.get("return_on_equity") is not None and metrics["return_on_equity"] > 0.15:  # Strong ROE above 15%
        profitability_score += 1
    if metrics.get("net_margin") is not None and metrics["net_margin"] > 0.20:  # Healthy profit margins
        profitability_score += 1
    if metrics.get("operating_margin") is not None and metrics["operating_margin"] > 0.15:  # Strong operating efficiency
        profitability_score += 1
        
    signals.append('bullish' if profitability_score >= 2 else 'bearish' if profitability_score == 0 else 'neutral')
    reasoning["Profitability"] = {
        "signal": signals[0],
        "details": (
            f"ROE: {format_metric(metrics.get('return_on_equity'))}, "
            f"Net Margin: {format_metric(metrics.get('net_margin'))}, "
            f"Op Margin: {format_metric(metrics.get('operating_margin'))}"
        )
    }
    
    # 2. Growth Analysis
    growth_score = 0
    if metrics.get("revenue_growth") is not None and metrics["revenue_growth"] > 0.10:  # 10% revenue growth
        growth_score += 1
    if metrics.get("earnings_growth") is not None and metrics["earnings_growth"] > 0.10:  # 10% earnings growth
        growth_score += 1
        
    signals.append('bullish' if growth_score >= 1 else 'bearish' if growth_score == 0 else 'neutral')
    reasoning["Growth"] = {
        "signal": signals[1],
        "details": (
            f"Revenue Growth: {format_metric(metrics.get('revenue_growth'))}, "
            f"Earnings Growth: {format_metric(metrics.get('earnings_growth'))}"
        )
    }
    
    # 3. Financial Health
    health_score = 0
    if metrics.get("current_ratio") is not None and metrics["current_ratio"] > 1.5:  # Strong liquidity
        health_score += 1

    if metrics.get("debt_to_equity") is not None and metrics["debt_to_equity"] < 0.5:  # Conservative debt levels
        health_score += 1

    if (metrics.get("free_cash_flow_per_share") is not None and 
        metrics.get("earnings_per_share") is not None and 
        metrics["free_cash_flow_per_share"] > metrics["earnings_per_share"] * 0.8):  # Strong FCF conversion
        health_score += 1
        
    signals.append('bullish' if health_score >= 2 else 'bearish' if health_score == 0 else 'neutral')
    reasoning["Financial_Health"] = {
        "signal": signals[2],
        "details": f"Current Ratio: {format_metric(metrics.get('current_ratio'))}, D/E: {format_metric(metrics.get('debt_to_equity'))}"
    }
    
    # 4. Valuation
    pe_ratio = metrics["price_to_earnings_ratio"] or 0
    pb_ratio = metrics["price_to_book_ratio"] or 0
    ps_ratio = metrics["price_to_sales_ratio"] or 0
    
    valuation_score = 0
    if pe_ratio is not None and pe_ratio < 25:  # Reasonable P/E ratio
        valuation_score += 1
    if pb_ratio is not None and pb_ratio < 3:  # Reasonable P/B ratio
        valuation_score += 1
    if ps_ratio is not None and ps_ratio < 5:  # Reasonable P/S ratio
        valuation_score += 1
        
    signals.append('bullish' if valuation_score >= 2 else 'bearish' if valuation_score == 0 else 'neutral')
    reasoning["Valuation"] = {
        "signal": signals[3],
        "details": f"P/E: {format_metric(pe_ratio)}, P/B: {format_metric(pb_ratio)}, P/S: {format_metric(ps_ratio)}"
    }
    
    # Determine overall signal
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    
    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    # Calculate confidence level
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals
    
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": reasoning
    }
    
    # Create the fundamental analysis message
    message = HumanMessage(
        content=str(message_content),
        name="fundamentals_agent",
    )
    
    reasoning_output = show_agent_reasoning(message_content) if show_reasoning else None
        
    return {
        "messages": state["messages"] + [message],
        "data": data,
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "fundamentals_agent": reasoning_output
            }
    }

##### Valuation Agent #####

def valuation_agent(state: AgentState):
    """Performs detailed valuation analysis using multiple methodologies."""
    data = state["data"]
    show_reasoning = state["metadata"]["show_reasoning"]
    valuation_config = state["metadata"].get("valuation_config", {})
    valuation_engine = valuation_config.get("engine", "v1")

    # Fetch the financial metrics
    metrics = get_financial_metrics(
        ticker=data["ticker"]
    )

    if valuation_engine == "v2":
        country = (metrics.get("country") or "US").upper().strip()
        selection = deterministic_method_fallback(
            ticker=data["ticker"],
            metrics=metrics,
            country=country,
            asset_type="",
        )
        if os.environ.get("GOOGLE_API_KEY"):
            try:
                selection = propose_method_and_assumptions_with_llm(
                    ticker=data["ticker"],
                    metrics=metrics,
                    country=country,
                )
            except Exception:
                selection = deterministic_method_fallback(
                    ticker=data["ticker"],
                    metrics=metrics,
                    country=country,
                    asset_type="",
                )

        payload = run_valuation(
            ticker=data["ticker"],
            metrics=metrics,
            asset_type=selection.get("asset_type", "stable-cashflow"),
            country=country,
            margin_of_safety_pct=0.25,
            llm_assumptions=selection.get("assumptions", {}),
            method_override=selection.get("valuation_method"),
            method_selection_reason=selection.get("valuation_method_reason"),
            method_selected_by=selection.get("selected_by", "deterministic_fallback"),
            method_selection_trace=selection.get("method_selection_trace", []),
            assumption_source="online",
            strict_mode=True,
            strict_mode_policy="autocorrect",
            monte_carlo_trials=500,
        )
        upside = payload["result"]["upside_downside_pct"] / 100.0
        signal = "bullish" if upside > 0.15 else "bearish" if upside < -0.15 else "neutral"
        message_content = {
            "signal": signal,
            "confidence": f"{min(abs(upside), 1.0):.0%}",
            "reasoning": {
                "details": payload,
            },
        }
        message = HumanMessage(
            content=json.dumps(message_content),
            name="valuation_agent",
        )
        reasoning_output = show_agent_reasoning(message.content) if show_reasoning else None
        return {
            "messages": state["messages"] + [message],
            "data": data,
            "metadata": {
                "valuation_model_used": (
                    f"valuation_v2_{payload.get('valuation_method', 'unknown')}"
                    f"_{payload.get('method_selection', {}).get('selected_by', 'unknown')}"
                ),
                "valuation_snapshot": {
                    "valuation_method": payload.get("valuation_method"),
                    "intrinsic_value_per_share": payload.get("result", {}).get("intrinsic_value_per_share"),
                    "current_price": payload.get("result", {}).get("current_price"),
                    "upside_downside_pct": payload.get("result", {}).get("upside_downside_pct"),
                    "margin_of_safety_price": payload.get("result", {}).get("margin_of_safety_price"),
                },
            },
            "agent_reasoning": {
                **state.get("agent_reasoning", {}),
                "valuation_agent": reasoning_output
            },
        }

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
                f"""Based on the company below, provide the data for the valuation model. Make your best estimate and assumptions.
                    Company: {data["ticker"]}
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
            "ticker": data["ticker"],
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

    try:
        result, model_used = invoke_gemini(prompt, temperature=0.1, max_tokens=None, max_retries=6, stop=None)
        result_content = result.content
    except Exception as exc:
        model_used = "deterministic_fallback"
        result_content = json.dumps(
            {
                "error": f"llm_unavailable: {exc}",
                "revenue_cagr": 0.05,
                f"profit_margin_{target_year}": 0.15,
                "shares_reduction_per_year": 0.01,
                "discount_rate": 0.1,
                "terminal_growth_rate": 0.03,
                "valuation_method": "PE_multiple",
                f"pe_ratio_{target_year}": 15,
            }
        )
    try:
        estimation = parse_output_to_json(result_content)
    except json.JSONDecodeError as e:
        estimation = {"error": f"JSON Decode Error {e}", "result": result_content}

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

    intrinsic_value_per_share = intrinsic_value / shares_value if shares_value else 0.0

    gap = min(((intrinsic_value_per_share - current_price) / current_price), 1.0) if current_price else 0.0
    signal = "bullish" if gap > 0.15 else "bearish" if gap < -0.15 else "neutral"

    reasoning = {       
        "details": {
            "intrinsic_value_per_share": round(intrinsic_value_per_share, 2),
            "estimated_metrics": estimation,
            "ltm_revenue": ltm_revenue,
            "shares_outstanding": shares_outstanding,
            "trailing_pe": trailing_pe,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "current_profit_margin": current_profit_margin,
        },
    }

    message_content = {
        "signal": signal,
        "confidence": f"{abs(gap):.0%}",
        "reasoning": reasoning
    }
    message = HumanMessage(
        content=json.dumps(message_content),
        name="valuation_agent",
    )

    reasoning_output = show_agent_reasoning(message.content) if show_reasoning else None

    return {"messages": state["messages"] + [message],
            "data": data,
            "metadata": {
                "valuation_model_used": model_used,
                "valuation_snapshot": {
                    "valuation_method": valuation_method,
                    "intrinsic_value_per_share": round(intrinsic_value_per_share, 2),
                    "current_price": current_price,
                    "upside_downside_pct": (
                        ((intrinsic_value_per_share / current_price) - 1.0) * 100.0 if current_price else 0.0
                    ),
                    "margin_of_safety_price": round(intrinsic_value_per_share * (1.0 - 0.25), 2),
                },
            },
            "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "valuation_agent": reasoning_output
            }}


##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Evaluates portfolio risk and sets position limits"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    prices = data["prices"]

    # Convert prices to a DataFrame
    prices_df = prices_to_df(prices)    

    quant_message = next(msg for msg in state["messages"] if msg.name == "quant_agent")
    fundamentals_message = next(msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    valuation_message = next(msg for msg in state["messages"] if msg.name == "valuation_agent")

    def _coerce_agent_payload(raw_content: str) -> dict:
        parsed = parse_output_to_json(raw_content)
        if not isinstance(parsed, dict):
            parsed = {}
        if "signal" not in parsed:
            parsed["signal"] = "neutral"
        if "confidence" not in parsed:
            parsed["confidence"] = "50%"
        return parsed

    fundamental_signals = _coerce_agent_payload(fundamentals_message.content)
    technical_signals = _coerce_agent_payload(quant_message.content)
    valuation_signals = _coerce_agent_payload(valuation_message.content)
        
    agent_signals = {
        "fundamental": fundamental_signals,
        "technical": technical_signals,
        "valuation": valuation_signals
    }

    # 1. Calculate Risk Metrics
    returns = prices_df['close'].pct_change().dropna()
    daily_vol = returns.std()
    volatility = daily_vol * (252 ** 0.5)  # Annualized volatility approximation
    var_95 = returns.quantile(0.05)         # Simple historical VaR at 95% confidence
    max_drawdown = (prices_df['close'] / prices_df['close'].cummax() - 1).min()

    # 2. Market Risk Assessment
    market_risk_score = 0

    # Volatility scoring
    if volatility.any() > 0.30:     # High volatility
        market_risk_score += 2
    elif volatility.any() > 0.20:   # Moderate volatility
        market_risk_score += 1

    # VaR scoring
    # Note: var_95 is typically negative. The more negative, the worse.
    if var_95.any() < -0.03:
        market_risk_score += 2
    elif var_95.any() < -0.02:
        market_risk_score += 1

    # Max Drawdown scoring
    if max_drawdown.any() < -0.20:  # Severe drawdown
        market_risk_score += 2
    elif max_drawdown.any() < -0.10:
        market_risk_score += 1

    # 3. Position Size Limits
    # Consider total portfolio value, not just cash
    current_stock_value = portfolio['invested'] * portfolio['stock']
    total_portfolio_value = portfolio['cash'] + portfolio['invested']
    base_position_size = total_portfolio_value * 0.25  # Start with 25% max position of total portfolio
    
    if market_risk_score >= 4:
        # Reduce position for high risk
        max_position_size = base_position_size * 0.5
    elif market_risk_score >= 2:
        # Slightly reduce for moderate risk
        max_position_size = base_position_size * 0.75
    else:
        # Keep base size for low risk
        max_position_size = base_position_size

    # 4. Stress Testing
    stress_test_scenarios = {
        "market_crash": -0.20,
        "moderate_decline": -0.10,
        "slight_decline": -0.05
    }

    stress_test_results = {}
    current_position_value = current_stock_value

    for scenario, decline in stress_test_scenarios.items():
        potential_loss = current_position_value * decline
        portfolio_impact = potential_loss / (portfolio['cash'] + current_position_value) if (portfolio['cash'] + current_position_value) != 0 else math.nan
        stress_test_results[scenario] = {
            "potential_loss": potential_loss,
            "portfolio_impact": portfolio_impact
        }
    
    # 5. Risk-Adjusted Signals Analysis
    # Convert all confidences to numeric for proper comparison
    def parse_confidence(conf_str):
        return float(conf_str.replace('%', '')) / 100.0
    low_confidence = any(parse_confidence(signal['confidence']) < 0.30 for signal in agent_signals.values())

    risk_score = (market_risk_score * 2)  # Market risk contributes up to ~6 points total when doubled
    if low_confidence:
        risk_score += 2  # Add penalty if any signal confidence < 30%   

    # Cap risk score at 10
    risk_score = min(round(risk_score), 10)

    # 6. Generate Trading Action
    if risk_score >= 8:
        trading_action = "reduce"
    elif risk_score >= 6:
        trading_action = "hold"
    else:
        trading_action = "add"

    # Calculate confidence scores for each metric
    def calculate_metric_confidence(metric, thresholds):
        """
        Calculate confidence score based on metric value and predefined thresholds
        Returns confidence percentage and assessment
        """
        if (abs(metric) <= thresholds['low']).all():
            return 90, "High confidence: Metric within normal range"
        elif (abs(metric) <= thresholds['medium']).all():
            return 70, "Moderate confidence: Metric showing some volatility"
        else:
            return 50, "Low confidence: Metric showing high volatility"

    # Define thresholds for each metric
    metric_thresholds = {
        'volatility': {'low': 0.15, 'medium': 0.25},
        'var': {'low': 0.02, 'medium': 0.03},
        'drawdown': {'low': 0.15, 'medium': 0.2}
    }

    # Calculate confidence for each metric
    volatility_conf, vol_assessment = calculate_metric_confidence(volatility, metric_thresholds['volatility'])
    var_conf, var_assessment = calculate_metric_confidence(abs(var_95), metric_thresholds['var'])
    drawdown_conf, dd_assessment = calculate_metric_confidence(abs(max_drawdown), metric_thresholds['drawdown'])

    # Calculate overall risk confidence
    confidence = round((volatility_conf + var_conf + drawdown_conf) / 3)

    # Update the message_content with confidence metrics
    message_content = {
        "max_position_size": float(max_position_size),
        "risk_score": risk_score,
        "trading_action": trading_action,
        "confidence": f"{confidence}%",
        "risk_metrics": {
            "volatility": {
                "value": float(volatility),
                "confidence": f"{volatility_conf}%",
                "assessment": vol_assessment
            },
            "value_at_risk_95": {
                "value": float(var_95),
                "confidence": f"{var_conf}%",
                "assessment": var_assessment
            },
            "max_drawdown": {
                "value": float(max_drawdown),
                "confidence": f"{drawdown_conf}%",
                "assessment": dd_assessment
            },
            "market_risk_score": market_risk_score,
            "stress_test_results": stress_test_results
        },
        "reasoning": f"Risk Score {risk_score}/10: Market Risk={market_risk_score}, "
                    f"Confidence={confidence}%, "
                    f"Volatility={float(volatility):.2%} (Conf: {volatility_conf}%), "
                    f"VaR={float(var_95):.2%} (Conf: {var_conf}%), "
                    f"Max Drawdown={float(max_drawdown):.2%} (Conf: {drawdown_conf}%)"
    }

    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_management_agent",
    )

    reasoning_output = show_agent_reasoning(message.content) if show_reasoning else None

    return {"messages": state["messages"] + [message],
            "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "risk_management_agent": reasoning_output
            }}


##### Portfolio Management Agent #####
def _build_portfolio_decision_fallback(
    *,
    quant_message: HumanMessage,
    fundamentals_message: HumanMessage,
    valuation_message: HumanMessage,
    risk_message: HumanMessage,
    portfolio: dict,
    market_news: dict,
) -> str:
    quant = parse_output_to_json(quant_message.content)
    fundamentals = parse_output_to_json(fundamentals_message.content)
    valuation = parse_output_to_json(valuation_message.content)
    risk = parse_output_to_json(risk_message.content)

    def _signal_value(payload: dict) -> int:
        signal = str(payload.get("signal", "neutral")).lower()
        if signal == "bullish":
            return 1
        if signal == "bearish":
            return -1
        return 0

    score = _signal_value(quant) + _signal_value(fundamentals) + _signal_value(valuation)
    risk_action = str(risk.get("trading_action", "hold")).lower()
    price = float(quant.get("current_price", 0.0) or 0.0)
    max_position_size = float(risk.get("max_position_size", 0.0) or 0.0)
    cash = float(portfolio.get("cash", 0.0) or 0.0)
    stock_weight = float(portfolio.get("stock", 0.0) or 0.0)

    action = "hold"
    if risk_action == "reduce":
        action = "sell" if stock_weight > 0 else "hold"
    elif risk_action == "hold":
        action = "hold"
    else:
        if score > 0 and cash > 0:
            action = "buy"
        elif score < 0 and stock_weight > 0:
            action = "sell"

    amount = 0.0
    quantity = 0.0
    if action == "buy" and price > 0:
        amount = min(max_position_size, cash * 0.2)
        quantity = amount / price if price else 0.0
    elif action == "sell" and price > 0:
        # No exact share count is tracked in this app; emit conservative placeholder.
        amount = 0.0
        quantity = 0.0

    confidence_pct = int(max(35, min(90, 50 + (score * 10))))
    agent_signals = [
        {"agent": "technical", "signal": quant.get("signal", "neutral"), "confidence": quant.get("confidence", "N/A")},
        {"agent": "fundamental", "signal": fundamentals.get("signal", "neutral"), "confidence": fundamentals.get("confidence", "N/A")},
        {"agent": "valuation", "signal": valuation.get("signal", "neutral"), "confidence": valuation.get("confidence", "N/A")},
    ]

    fallback_payload = {
        "price": round(price, 2),
        "action": action,
        "confidence": f"{confidence_pct}%",
        "amount": round(amount, 2),
        "quantity": round(quantity, 4),
        "agent_signals": agent_signals,
        "reasoning": (
            "Fallback deterministico activado por falla/capacidad del modelo. "
            f"Score agregado={score}, risk_action={risk_action}, max_position_size={max_position_size:.2f}."
        ),
        "news": market_news.get("answer", ""),
    }
    return json.dumps(fallback_payload)


def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    market_news = state["data"]["market_news"]
    logo_url = state["data"]["financial_metrics"]["logo_url"]

    # Get the quant agent, fundamentals agent, and risk management agent messages
    quant_message = next(msg for msg in state["messages"] if msg.name == "quant_agent")
    fundamentals_message = next(msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    risk_message = next(msg for msg in state["messages"] if msg.name == "risk_management_agent")
    valuation_message = next(msg for msg in state["messages"] if msg.name == "valuation_agent")

    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions.
                Your job is to make a trading decision based on the team's analysis.
                Add metric values to enrich the analysis.

                Provide the following in your output as json:
                - "price": <(current_price):.2f>
                - "action": <"buy" | "sell" | "hold">
                - "confidence": <percentage between 0 and 100%>
                - "amount": <(porfolio_cash * max_position_size / 100):.2f>
                - "quantity": <(amount / price)>
                - "agent_signals": <list of agent signals including agent name, signal (bullish | bearish | neutral), and their confidence>
                - "reasoning": <concise explanation of the decision including metrics and how you weighted the signals>
                
                Trading Rules:
                - Never exceed risk management position limits
                - Only buy if you have available cash
                - Only sell if you have shares to sell
                - When Hold, no amount or quantity is needed
                - Quantity must be ≤ current position for sells
                - Quantity must be ≤ max_position_size from risk management

                Add a paragraph on news of the stock only if are relevants to the company. 
                Don't use news for decision making. It's only for context.
                """
            ),
            (
                "human",
                """Based on the team's analysis below, make your trading decision.

                Quant Analysis Trading Signal: {quant_message}
                Fundamental Analysis Trading Signal: {fundamentals_message}
                Valuation Analysis Trading Signal: {valuation_message}
                Risk Management Trading Signal: {risk_message}

                Here is the current portfolio:
                Portfolio:
                Cash Available: {portfolio_cash}
                Current Position: {portfolio_stock} %

                News: 
                Answer: {answer_news}
                Links: {links}

                Only include the price, action, quantity, amount, reasoning and news in your output as JSON.  Do not include any JSON markdown.

                Remember, the action must be either buy, sell, or hold.
                You can only buy if you have available cash. Current price plus quantity must be less than cash.
                You can only sell if you have shares in the portfolio to sell. Sell if the gains are at risk.

                """
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "quant_message": quant_message.content,
            "fundamentals_message": fundamentals_message.content,
            "valuation_message": valuation_message.content,
            "risk_message": risk_message.content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"],
            "answer_news": market_news["answer"],
            "links": market_news["results"]
        }
    )
    # Invoke the LLM with deterministic fallback when model quota/capacity fails.
    try:
        result, model_used = invoke_gemini(prompt, temperature=0.1, max_tokens=None, max_retries=6, stop=None)
        message_content = result.content
    except Exception:
        model_used = "deterministic_fallback"
        message_content = _build_portfolio_decision_fallback(
            quant_message=quant_message,
            fundamentals_message=fundamentals_message,
            valuation_message=valuation_message,
            risk_message=risk_message,
            portfolio=portfolio,
            market_news=market_news,
        )

    # Create the portfolio management message
    message = HumanMessage(
        content=message_content,
        name="portfolio_management",
    )

    reasoning_output = show_agent_reasoning(message.content) if show_reasoning else None

    return {"messages": state["messages"] + [message],
            "metadata": {
                "logo_url": logo_url,
                "portfolio_model_used": model_used,
                },
            "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "portfolio_management": reasoning_output
            }
            }

def show_agent_reasoning(output):
    """
    Formatea el razonamiento de un agente en JSON.
    
    Args:
        output: La salida del agente (puede ser un diccionario, lista, cadena JSON, o cadena de texto)
        agent_name: Nombre del agente para mostrar en el encabezado
    """

    try:
        # Convertir la salida a un formato JSON que Streamlit pueda mostrar
        json_output = parse_output_to_json(output)
        
        # Mostrar el JSON en Streamlit
        return json_output
    
    except Exception as e:
        # Manejar cualquier error de conversión
        return {"error": str(e), "original_output": str(output)}


def classify_error(exc: Exception) -> str:
    message = str(exc).lower()
    if any(token in message for token in ["404", "not_found", "model", "gemini", "google"]):
        return "llm"
    if any(token in message for token in ["json", "parse", "decode"]):
        return "parse"
    if any(token in message for token in ["price", "ticker", "yfinance", "data"]):
        return "data_fetch"
    return "unknown"


##### Run the Hedge Fund #####
def run_hedge_fund(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    valuation_config: dict | None = None,
):
    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data."
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
            },
            "metadata": {
                "show_reasoning": show_reasoning,
                "valuation_config": valuation_config or {"engine": "v1"},
            }
        },
    )
    metadata = final_state.get("metadata", {})
    agent_reasoning = final_state.get("agent_reasoning", {})
    agent_reasoning["_meta"] = {
        "valuation_model_used": metadata.get("valuation_model_used"),
        "portfolio_model_used": metadata.get("portfolio_model_used"),
        "valuation_snapshot": metadata.get("valuation_snapshot"),
    }
    return final_state["messages"][-1].content, metadata.get("logo_url"), agent_reasoning

# Define the new workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("quant_agent", quant_agent)
workflow.add_node("fundamentals_agent", fundamentals_agent)
workflow.add_node("valuation_agent", valuation_agent)
workflow.add_node("risk_management_agent", risk_management_agent)
workflow.add_node("portfolio_management_agent", portfolio_management_agent)

# Define the workflow
workflow.set_entry_point("market_data_agent")
workflow.add_edge("market_data_agent", "quant_agent")
workflow.add_edge("market_data_agent", "fundamentals_agent")
workflow.add_edge("market_data_agent", "valuation_agent")
workflow.add_edge("quant_agent", "risk_management_agent")
workflow.add_edge("fundamentals_agent", "risk_management_agent")
workflow.add_edge("valuation_agent", "risk_management_agent")
workflow.add_edge("risk_management_agent", "portfolio_management_agent")
workflow.add_edge("portfolio_management_agent", END)

app = workflow.compile()

# Add this at the bottom of the file
if __name__ in {"__main__", "__page__"}:
    st.title("Portfolio AI Agents Analysis :material/finance_mode:")
    st.caption("Build your portfolio mix, then run multi-agent analysis per ticker.")

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {}

    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning(
            "`GOOGLE_API_KEY` is missing. LLM steps will use deterministic fallback where available."
        )

    with st.form("portfolio_builder", clear_on_submit=True):
        left, middle, right = st.columns([4, 4, 2], gap="large", vertical_alignment="bottom")
        with left:
            ticker_input = st.text_input("Ticker", max_chars=8, placeholder="AAPL").upper().strip()
        with middle:
            percentage_input = st.slider("Allocation (%)", 0.0, 100.0, 10.0, 1.0)
        with right:
            add_ticker = st.form_submit_button("Add")

        if add_ticker:
            if not ticker_input:
                st.warning("Enter a ticker before adding.")
            else:
                st.session_state.portfolio[ticker_input] = percentage_input / 100.0
                st.success(f"Added/updated `{ticker_input}` at {percentage_input:.1f}%.")

    manage_col1, manage_col2 = st.columns([3, 1], gap="large")
    with manage_col1:
        if st.session_state.portfolio:
            ticker_to_remove = st.selectbox("Remove ticker", options=list(st.session_state.portfolio.keys()))
            if st.button("Remove selected", width="stretch"):
                st.session_state.portfolio.pop(ticker_to_remove, None)
                st.rerun()
        else:
            st.info("No tickers added yet.")
    with manage_col2:
        if st.button("Clear portfolio", width="stretch"):
            st.session_state.portfolio = {}
            st.rerun()

    total_percentage = 0.0
    if st.session_state.portfolio:
        portfolio_data = []
        for ticker, percentage in st.session_state.portfolio.items():
            pct = float(percentage * 100)
            total_percentage += pct
            portfolio_data.append({"Ticker": ticker, "Allocation (%)": pct})
        st.subheader("Current Portfolio")
        st.dataframe(portfolio_data, width="stretch", hide_index=True)
        st.metric("Total Allocation", f"{total_percentage:.2f}%")
        if total_percentage > 100:
            st.error("Total allocation is above 100%. Reduce weights before running analysis.")

    with st.expander("Analysis settings", expanded=True):
        left, middle, right = st.columns(3)
        with left:
            cash = st.number_input("Cash available (USD)", min_value=0.0, value=5000.0, step=500.0)
        with middle:
            invested = st.number_input(
                "Current invested value (USD)", min_value=0.0, value=20000.0, step=1000.0
            )
        with right:
            show_reasoning = st.checkbox("Show reasoning from each agent", value=False)
        v1_col, v2_col = st.columns(2)
        with v1_col:
            valuation_engine = st.selectbox(
                "Valuation Engine",
                options=["v1", "v2"],
                index=0,
                help="v1: legacy AI-assisted DCF. v2: automatic method selection + deterministic valuation.",
            )
        with v2_col:
            if valuation_engine == "v2":
                st.caption("V2 defaults: auto method selection, country inferred, 25% MoS, strict checks on.")
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        with col_end:
            end_date = st.date_input("End Date", value=datetime.now())

    if start_date > end_date:
        st.error("Start date must be earlier than end date.")

    run_disabled = (
        not st.session_state.portfolio
        or total_percentage <= 0
        or total_percentage > 100
        or start_date > end_date
    )

    if st.button("Run Agents Analysis", type="primary", disabled=run_disabled):
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        valuation_config = {
            "engine": valuation_engine,
        }
        items = list(st.session_state.portfolio.items())
        progress = st.progress(0, text="Starting analysis...")

        success_count = 0
        warning_count = 0
        error_count = 0

        for idx, (ticker, weight) in enumerate(items, start=1):
            details = {"cash": cash, "stock": weight, "invested": invested}
            progress.progress(int((idx - 1) / len(items) * 100), text=f"Analyzing {ticker}...")

            try:
                result, logo_url, agent_reasoning = run_hedge_fund(
                    ticker=ticker,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    portfolio=details,
                    show_reasoning=show_reasoning,
                    valuation_config=valuation_config,
                )
            except Exception as exc:
                error_type = classify_error(exc)
                st.error(f"[{error_type}] Error running agent for {ticker}: {exc}")
                error_count += 1
                continue

            parsed = parse_output_to_json(result)
            summary = extract_summary_fields(parsed)
            meta = agent_reasoning.get("_meta", {}) if isinstance(agent_reasoning, dict) else {}
            success_count += 1
            with st.expander(f"{ticker} analysis", expanded=True):
                col1, col2 = st.columns([1, 9], gap="small", vertical_alignment="center")
                with col1:
                    if logo_url:
                        try:
                            st.image(logo_url, width=90)
                        except Exception:
                            warning_count += 1
                            st.caption("Logo unavailable")
                    else:
                        warning_count += 1
                        st.caption("Ticker without logo")
                with col2:
                    st.markdown(f"**[{ticker}](https://finviz.com/quote.ashx?t={ticker}&p=d)**")
                    valuation_model = meta.get("valuation_model_used") or "unknown"
                    portfolio_model = meta.get("portfolio_model_used") or "unknown"
                    st.caption(
                        f"Models used: valuation=`{valuation_model}` | portfolio_manager=`{portfolio_model}`"
                    )

                chart_url = build_finviz_chart_url(ticker)
                try:
                    st.image(chart_url, width="stretch")
                except Exception:
                    warning_count += 1
                    st.caption(
                        f"Chart unavailable. [Open Finviz](https://finviz.com/quote.ashx?t={ticker}&p=d)"
                    )

                tab_summary, tab_raw = st.tabs(["Executive Summary", "Raw JSON"])
                with tab_summary:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Action", str(summary["action"]))
                    m2.metric("Confidence", str(summary["confidence"]))
                    m3.metric("Price", str(summary["price"]))
                    m4.metric("Amount", str(summary["amount"]))
                    m5.metric("Quantity", str(summary["quantity"]))

                    valuation_snapshot = meta.get("valuation_snapshot") if isinstance(meta, dict) else None
                    if isinstance(valuation_snapshot, dict):
                        intrinsic = valuation_snapshot.get("intrinsic_value_per_share")
                        current_px = valuation_snapshot.get("current_price")
                        upside = valuation_snapshot.get("upside_downside_pct")
                        mos_price = valuation_snapshot.get("margin_of_safety_price")
                        method = valuation_snapshot.get("valuation_method")

                        v1, v2, v3, v4 = st.columns(4)
                        v1.metric(
                            "Intrinsic Value",
                            f"${float(intrinsic):.2f}" if isinstance(intrinsic, (int, float)) else "N/A",
                        )
                        v2.metric(
                            "Current Price (Valuation)",
                            f"${float(current_px):.2f}" if isinstance(current_px, (int, float)) else "N/A",
                        )
                        v3.metric(
                            "Upside/Downside",
                            f"{float(upside):.2f}%" if isinstance(upside, (int, float)) else "N/A",
                        )
                        v4.metric(
                            "MoS Price",
                            f"${float(mos_price):.2f}" if isinstance(mos_price, (int, float)) else "N/A",
                        )
                        if method:
                            st.caption(f"Valuation method used: `{method}`")

                    st.markdown(f"**Reasoning:** {summary['reasoning']}")
                    if summary.get("news"):
                        st.markdown(f"**News context:** {summary['news']}")
                    if summary["agent_signals"]:
                        st.markdown("**Agent Signals**")
                        st.dataframe(summary["agent_signals"], width="stretch", hide_index=True)
                with tab_raw:
                    st.json(parsed)

                if show_reasoning:
                    st.markdown("#### Agent Reasonings")
                    for agent, reasoning in agent_reasoning.items():
                        if agent == "_meta":
                            continue
                        st.markdown(f"**{agent.replace('_', ' ').title()}**")
                        st.json(reasoning)

        progress.progress(100, text="Analysis completed.")
        st.success(f"Completed: {success_count} ticker(s) analyzed.")
        if warning_count:
            st.warning(f"{warning_count} render warning(s) detected (missing logo/chart).")
        if error_count:
            st.warning(f"{error_count} ticker(s) failed during execution.")
