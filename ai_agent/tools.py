import os

import pandas as pd
import requests
import yfinance as yf
from tavily import TavilyClient
  
def get_prices(ticker, start_date, end_date):
    """Fetch price data from the API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}"
        f"&interval=day"
        f"&interval_multiplier=1"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    prices = data.get("prices")
    if not prices:
        raise ValueError("No price data returned")
    return prices

def prices_to_df(prices):
    """Convert prices to a DataFrame."""
    df = pd.DataFrame(prices)
    # df = df.drop(columns=["Close"])
    df = df.rename(
        columns={
            "Open": "open",
            "Adj Close": "close",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
        }
    )
    df.sort_index(inplace=True)
    return df

# Update the get_price_data function to use the new functions
def get_price_data(ticker, start_date, end_date):
    prices = yf.download(ticker, start=start_date, end=end_date)
    return prices_to_df(prices)

def calculate_confidence_level(signals):
    """Calculate confidence level based on the difference between SMAs."""
    sma_diff_prev = abs(signals['sma_5_prev'] - signals['sma_20_prev'])
    sma_diff_curr = abs(signals['sma_5_curr'] - signals['sma_20_curr'])
    diff_change = sma_diff_curr - sma_diff_prev
    # Normalize confidence between 0 and 1
    confidence = min(max(diff_change / signals['current_price'], 0), 1)
    return confidence

def calculate_macd(prices_df):
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_rsi(prices_df, period=14):
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices_df, window=20):
    sma = prices_df['close'].rolling(window).mean()
    std_dev = prices_df['close'].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_obv(prices_df):
    obv = [0]
    for i in range(1, len(prices_df)):
        if prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] + prices_df['volume'].iloc[i])
        elif prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]:
            obv.append(obv[-1] - prices_df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    prices_df['OBV'] = obv
    return prices_df['OBV']

def get_financial_metrics(ticker):
    """
    Fetch financial metrics using yfinance.
    Returns:
        A dictionary with financial metrics:
        return_on_equity, net_margin, operating_margin, revenue_growth, 
        earnings_growth, book_value_growth, current_ratio, debt_to_equity, 
        free_cash_flow_per_share, earnings_per_share, price_to_earnings_ratio, 
        price_to_book_ratio, price_to_sales_ratio.
    """
    stock = yf.Ticker(ticker)
    info = stock.info  # Información financiera general
    
    try:
        financial_metrics = {
            "return_on_equity": info.get("returnOnEquity"),
            "net_margin": info.get("netIncomeToCommon") / info.get("totalRevenue") if info.get("totalRevenue") else None,
            "operating_margin": info.get("operatingMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "current_ratio": info.get("currentRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cash_flow_per_share": info.get("freeCashflow") / info.get("sharesOutstanding") if info.get("sharesOutstanding") else None,
            "earnings_per_share": info.get("trailingEps"),
            "price_to_earnings_ratio": info.get("trailingPE"),
            "price_to_book_ratio": info.get("priceToBook"),
            "price_to_sales_ratio": info.get("priceToSalesTrailing12Months"),
        }
    except KeyError as e:
        raise ValueError(f"Error fetching data for {ticker}: {e}")
    
    return financial_metrics

def format_metric(metric):
    if metric is None:
        return "N/A"
    else:
        return f"{metric:.2%}"

def get_news(
    query: str,
    days:  int = 20,
    max_results: int = 5,
):
    """
    Perform a web search using the Tavily API.

    This tool accesses real-time web data, news, articles and should be used when up-to-date information from the internet is required.
    """
    exclude_domains = ["sports.yahoo.com", "marca.com"]

    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    response = client.search(
        query, 
        topic="news", 
        days=days, 
        max_results=max_results, 
        include_answer=True, 
        exclude_domains=exclude_domains)

    new_response = {
        "answer": response['answer'],
        "results": response['results']
    }
    
    
    return new_response    