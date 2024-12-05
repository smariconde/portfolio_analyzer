from typing import Annotated, Any, Dict, Sequence, TypedDict

import operator, os, json, re, time
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from tools import calculate_bollinger_bands, calculate_macd, calculate_obv, calculate_rsi, get_price_data, prices_to_df

import streamlit as st
from datetime import datetime, timedelta



if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ.get('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-002",
    temperature=0,
    max_tokens=None,
    max_retries=6,
    stop=None
)

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Dict[str, Any]

##### 1. Market Data Agent #####
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

    return {
        "messages": messages,
        "data": {**data, "prices": prices, "start_date": start_date, "end_date": end_date}
    }

##### 2. Quantitative Agent #####
def quant_agent(state: AgentState):
    """Analyzes technical indicators and generates trading signals."""
    show_reasoning = state["messages"][0].additional_kwargs["show_reasoning"]

    data = state["data"]
    prices = data["prices"]
    prices_df = prices_to_df(prices)
    
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

    # Create the quant message
    message = HumanMessage(
        content=str(message_content),  # Convert dict to string for message content
        name="quant_agent",
    )

    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Quant Agent")
    
    return {
        "messages": state["messages"] + [message],
        "data": data
    }

##### 3. Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Evaluates portfolio risk and sets position limits"""
    show_reasoning = state["messages"][0].additional_kwargs["show_reasoning"]
    portfolio = state["messages"][0].additional_kwargs["portfolio"]
    quant_message = state["messages"][-1]

    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a risk management specialist.
                Your job is to take a look at the trading analysis and
                evaluate portfolio exposure and recommend the percentage position sizing.
                Provide the following in your output (as a JSON):
                "max_position_size": <float between 1 and 100>,
                "risk_score": <integer between 1 and 10>,
                "trading_action": <buy | sell | hold>,
                "reasoning": <concise explanation of the decision>
                Max position size must be the maximum percentage of the portfolio of this stock.
                """
            ),
            (
                "human",
                """Based on the trading analysis below, provide your risk assessment.

                Quant Trading Signal: {quant_message}

                Here is the current portfolio:
                Portfolio:
                Cash Available: {portfolio_cash}
                Current Position: {portfolio_stock} % 
                
                Only include the max position size, risk score, trading action, and reasoning in your JSON output.
                Do not include any JSON markdown.
                """
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "quant_message": quant_message.content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"]
        }
    )

    # Invoke the LLM
    result = llm.invoke(prompt)
    message = HumanMessage(
        content=result.content,
        name="risk_management",
    )

    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Risk Management Agent")

    return {"messages": state["messages"] + [message]}


##### 4. Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders"""
    show_reasoning = state["messages"][0].additional_kwargs["show_reasoning"]
    portfolio = state["messages"][0].additional_kwargs["portfolio"]
    risk_message = state["messages"][-1]
    quant_message = state["messages"][-2]

    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions.
                Your job is to make a trading decision based on the team's analysis.
                Provide the following in your output as json:
                - "price": <Current price>
                - "action": "buy" | "sell" | "hold",
                - "quantity": <positive integer>
                - "amount": <(current price * (cash * max_position_size / 100)):.2f>
                - "reasoning": <concise explanation of the decision>
                Only buy if you have available cash.
                The quantity that you buy must be less than or equal to the max position size percentage.
                Only sell if you have shares in the portfolio to sell.
                The quantity that you sell must be less than or equal to the current position size percentage.
                """
            ),
            (
                "human",
                """Based on the team's analysis below, make your trading decision.

                Quant Team Trading Signal: {quant_message}
                Risk Management Team Signal: {risk_message}

                Here is the current portfolio:
                Portfolio:
                Cash Available: {portfolio_cash}
                Current Position: {portfolio_stock} %

                Only include the price, action, quantity, amount and reasoning in your output as JSON.  Do not include any JSON markdown.

                Remember, the action must be either buy, sell, or hold.
                You can only buy if you have available cash. Current price plus quantity must be less tahn cash.
                You can only sell if you have shares in the portfolio to sell.
                """
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "quant_message": quant_message.content, 
            "risk_message": risk_message.content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"]
        }
    )
    # Invoke the LLM
    result = llm.invoke(prompt)

    # Create the portfolio management message
    message = HumanMessage(
        content=result.content,
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}

def show_agent_reasoning(output, agent_name):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")
    st.markdown(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")
    if isinstance(output, (dict, list)):
        # If output is already a dictionary or list, just pretty print it
        print(json.dumps(output, indent=2))
        st.json(json.dumps(output, indent=2))
    else:
        try:
            # Parse the string as JSON and pretty print it

            cleaned_output = re.sub(r"\s*```json\s*", '', output).strip()
            cleaned_output = re.sub(r"\```", '"', cleaned_output)  # Replace single quotes with double quotes
            cleaned_output = re.sub(r'\s+', ' ', cleaned_output)  # Replace multiple spaces with a single space
            output_match = re.search(r'(\{.*\})', cleaned_output)
            st.json(json.loads(output_match.group(1)))
        except json.JSONDecodeError:
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
            st.markdown(parsed_output)
    print("=" * 48)


##### Run the Hedge Fund #####
def run_hedge_fund(ticker: str, start_date: str, end_date: str, portfolio: dict, show_reasoning: bool = False):
    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                    additional_kwargs={
                        "portfolio": portfolio,
                        "show_reasoning": show_reasoning,
                    },
                )
            ],
            "data": {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date
            },
        },
    )
    return final_state["messages"][-1].content

# Define the new workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("quant_agent", quant_agent)
workflow.add_node("risk_management_agent", risk_management_agent)
workflow.add_node("portfolio_management_agent", portfolio_management_agent)

# Define the workflow
workflow.set_entry_point("market_data_agent")
workflow.add_edge("market_data_agent", "quant_agent")
workflow.add_edge("quant_agent", "risk_management_agent")
workflow.add_edge("risk_management_agent", "portfolio_management_agent")
workflow.add_edge("portfolio_management_agent", END)

app = workflow.compile()

# Add this at the bottom of the file
if __name__ == "__main__":
    st.title("Portfolio AI Agents Analysis")

    # Create input fields for user inputs
    portfolio_input = st.text_area('Enter Portfolio (comma-separated ticker:percentage pairs)', 'AAPL:10%,GOOGL:20%,MSFT:70%')
    cash = st.number_input("Cash available", value=10000)
    portfolio = {}
    items = portfolio_input.split(',')
    for item in items:
        parts = item.strip().split(':')
        if len(parts) != 2:
            st.error(f"Invalid format for '{item}'. Expected 'ticker:percentage'.")
            continue
        
        ticker, percentage = parts
        try:
            percentage = float(percentage.strip('%'))
        except ValueError:
            st.error(f"Invalid percentage value for '{ticker}'. Please enter a valid number.")
            continue
        
        if not (0 <= percentage <= 100):
            st.error(f"Percentage for '{ticker}' must be between 0 and 100.")
            continue
        portfolio[ticker.strip()] = {"cash": cash, "stock": percentage}

    start_date = st.date_input(f'Start Date', value=datetime.now() - timedelta(days=90))
    end_date = st.date_input(f'End Date', value=datetime.now())
    show_reasoning = st.checkbox(f'Show Reasoning from Each Agent')

    # Convert dates to string format for validation
    start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
    end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None

    # Validate dates if provided
    if start_date_str:
        try:
            datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            st.error(f"Start date must be in YYYY-MM-DD format")
    
    if end_date_str:
        try:
            datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            st.error(f"End date must be in YYYY-MM-DD format")

    # Button to run the hedge fund
    if st.button('Run Hedge Fund'):
        for ticker, details in portfolio.items():
            result = run_hedge_fund(
                ticker=ticker,
                start_date=start_date_str,
                end_date=end_date_str,
                portfolio=details,
                show_reasoning=show_reasoning
            )

            # Remove occurrences of '''json from the result and clean up
            cleaned_result = re.sub(r"\s*```json\s*", '', result).strip()
            cleaned_result = re.sub(r"\```", '"', cleaned_result)  # Replace single quotes with double quotes
            cleaned_result = re.sub(r'\s+', ' ', cleaned_result)  # Replace multiple spaces with a single space
            json_match = re.search(r'(\{.*\})', cleaned_result) 

            # Check if cleaned_result is not empty and is valid JSON
            if json_match:
                try:
                    st.markdown(f"**[{ticker}](https://finviz.com/quote.ashx?t={ticker}&p=d)**")
                    json_result = json.loads(json_match.group(1))  # Parse the result
                    st.json(json_result)  # Display the JSON result
                except json.JSONDecodeError:
                    st.error("The AI agent returned an invalid JSON response.")
            else:
                st.error("The AI agent returned an empty response.")

            time.sleep(8)