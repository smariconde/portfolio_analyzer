import os, math, json, re
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tavily import TavilyClient
from typing import Dict

from ai_agent.env_loader import load_project_env

load_project_env()

def parse_output_to_json(output):
    """Parses a string or object into a JSON dictionary."""
    if isinstance(output, (dict, list)):
        return output
    if isinstance(output, str):
        cleaned_output = re.sub(r"```(json)?", '', output).strip()
        try:
            return json.loads(cleaned_output)
        except json.JSONDecodeError:
            try:
                match = re.search(r'(\{.*\})', cleaned_output, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
            except:
                pass
    return {"raw_output": str(output)}

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
    if "Adj Close" in df.columns:
        df = df.rename(
            columns={
                "Open": "open",
                "Adj Close": "close",
                "High": "high",
                "Low": "low",
                "Volume": "volume",
            }
        )
    elif "Close" in df.columns:
        df = df.rename(
            columns={
                "Open": "open",
                "Close": "close",
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
        if np.where(prices_df['close'].iloc[i] > prices_df['close'].iloc[i - 1]):
            obv.append(obv[-1] + prices_df['volume'].iloc[i])
        elif np.where(prices_df['close'].iloc[i] < prices_df['close'].iloc[i - 1]):
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
            "debt_to_equity": info.get("debtToEquity") / 100 if info.get("debtToEquity") else None,
            "free_cash_flow_per_share": info.get("freeCashflow") / info.get("sharesOutstanding") if (info.get("sharesOutstanding") and info.get("freeCashFlow")) else None,
            "earnings_per_share": info.get("trailingEps"),
            "price_to_earnings_ratio": info.get("trailingPE"),
            "price_to_book_ratio": info.get("priceToBook"),
            "price_to_sales_ratio": info.get("priceToSalesTrailing12Months"),
            "totalRevenue": info.get("totalRevenue"),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "currentPrice": info.get("currentPrice"),
            "profitMargins": info.get("profitMargins"),
            "industry": info.get("industry"),
            "sector": info.get("sector"),
            "targetMeanPrice": info.get("targetMeanPrice"),
            "logo_url": f"https://logo.clearbit.com/{info.get('website', '').replace('http://', '').replace('https://', '').split('/')[0]}"
        }
    except KeyError as e:
        raise ValueError(f"Error fetching data for {ticker}: {e}")
    
    return financial_metrics

def format_metric(metric):
    if metric is None:
        return "N/A"
    else:
        return f"{metric:.2f}"

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


def calculate_trend_signals(prices_df):
    """
    Advanced trend following strategy using multiple timeframes and indicators
    """
    # Calculate EMAs for multiple timeframes
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # Calculate ADX for trend strength
    adx = calculate_adx(prices_df, 14)

    # Determine trend direction and strength
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # Combine signals with confidence weighting
    trend_strength = adx['adx'].iloc[-1] / 100.0

    if short_trend.iloc[-1].all() and medium_trend.iloc[-1].all():
        signal = 'bullish'
        confidence = trend_strength
    elif not short_trend.iloc[-1].any() and not medium_trend.iloc[-1].any():
        signal = 'bearish'
        confidence = trend_strength
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'adx': float(adx['adx'].iloc[-1]),
            'trend_strength': float(trend_strength),
            # 'ichimoku': ichimoku
        }
    }

def calculate_mean_reversion_signals(prices_df):
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # Calculate z-score of price relative to moving average
    ma_50 = prices_df['close'].rolling(window=50).mean()
    std_50 = prices_df['close'].rolling(window=50).std()
    z_score = (prices_df['close'] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    extreme_z_score = abs(z_score.iloc[-1]) > 2
    price_vs_bb = (prices_df['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Combine signals
    if (z_score.iloc[-1] < -2).any() and (price_vs_bb < 0.2).any():
        signal = 'bullish'
        confidence = min(abs(z_score.iloc[-1].any()) / 4, 1.0)
    elif (z_score.iloc[-1] > 2).any() and (price_vs_bb > 0.8).any():
        signal = 'bearish'
        confidence = min(abs(z_score.iloc[-1].any()) / 4, 1.0)
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'z_score': float(z_score.iloc[-1]),
            'price_vs_bb': float(price_vs_bb),
            'rsi_14': float(rsi_14.iloc[-1]),
            'rsi_28': float(rsi_28.iloc[-1])
        }
    }

def calculate_momentum_signals(prices_df):
    """
    Multi-factor momentum strategy
    """
    # Price momentum
    returns = prices_df['close'].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    # Volume momentum
    volume_ma = prices_df['volume'].rolling(21).mean()
    volume_momentum = prices_df['volume'] / volume_ma

    # Relative strength
    # (would compare to market/sector in real implementation)

    # Calculate momentum score
    momentum_score = (
        0.4 * mom_1m +
        0.3 * mom_3m +
        0.3 * mom_6m
    ).iloc[-1]

    # Volume confirmation
    volume_confirmation = volume_momentum.iloc[-1].any() > 1.0

    if (momentum_score > 0.05).any() and volume_confirmation:
        signal = 'bullish'
        confidence = min(abs(momentum_score).max() * 5, 1.0)
    elif (momentum_score < -0.05).any() and volume_confirmation:
        signal = 'bearish'
        confidence = min(abs(momentum_score).max() * 5, 1.0)
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'momentum_1m': float(mom_1m.iloc[-1]),
            'momentum_3m': float(mom_3m.iloc[-1]),
            'momentum_6m': float(mom_6m.iloc[-1]),
            'volume_momentum': float(volume_momentum.iloc[-1])
        }
    }

def calculate_volatility_signals(prices_df):
    """
    Volatility-based trading strategy
    """
    # Calculate various volatility metrics
    returns = prices_df['close'].pct_change()

    # Historical volatility
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # Volatility regime detection
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    # Volatility mean reversion
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    # ATR ratio
    # atr = calculate_atr(prices_df)
    # atr_ratio = atr / prices_df['close']

    # Generate signal based on volatility regime
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if (current_vol_regime < 0.8).any() and (vol_z < -1).any():
        signal = 'bullish'  # Low vol regime, potential for expansion
        confidence = min(abs(vol_z).max() / 3, 1.0)
    elif (current_vol_regime > 1.2).any() and (vol_z > 1).any():
        signal = 'bearish'  # High vol regime, potential for contraction
        confidence = min(abs(vol_z).max() / 3, 1.0)
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'historical_volatility': float(hist_vol.iloc[-1]),
            'volatility_regime': float(current_vol_regime),
            'volatility_z_score': float(vol_z),
            # 'atr_ratio': atr_ratio.iloc[-1]
        }
    }

def calculate_stat_arb_signals(prices_df):
    """
    Statistical arbitrage signals based on price action analysis
    """
    # Calculate price distribution statistics
    returns = prices_df['close'].pct_change()

    # Skewness and kurtosis
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()

    # Test for mean reversion using Hurst exponent
    hurst = calculate_hurst_exponent(prices_df['close'])

    # Correlation analysis
    # (would include correlation with related securities in real implementation)

    # Generate signal based on statistical properties
    if hurst < 0.4 and (skew.iloc[-1] > 1).any():
        signal = 'bullish'
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and (skew.iloc[-1] < -1).any():
        signal = 'bearish'
        confidence = (0.5 - hurst) * 2
    else:
        signal = 'neutral'
        confidence = 0.5

    return {
        'signal': signal,
        'confidence': confidence,
        'metrics': {
            'hurst_exponent': float(hurst),
            'skewness': float(skew.iloc[-1]),
            'kurtosis': float(kurt.iloc[-1])
        }
    }

def weighted_signal_combination(signals, weights):
    """
    Combines multiple trading signals using a weighted approach
    """
    # Convert signals to numeric values
    signal_values = {
        'bullish': 1,
        'neutral': 0,
        'bearish': -1
    }

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        numeric_signal = signal_values[signal['signal']]
        weight = weights[strategy]
        confidence = signal['confidence']

        weighted_sum += numeric_signal * weight * confidence
        total_confidence += weight * confidence

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Convert back to signal
    if final_score > 0.2:
        signal = 'bullish'
    elif final_score < -0.2:
        signal = 'bearish'
    else:
        signal = 'neutral'

    return {
        'signal': signal,
        'confidence': abs(total_confidence),
    }

def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def calculate_macd(prices_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line

def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        df: DataFrame with price data
        window: EMA period
    
    Returns:
        pd.Series: EMA values
    """
    return df['close'].ewm(span=window, adjust=False).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)
    
    Args:
        df: DataFrame with OHLC data
        period: Period for calculations
    
    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # Calculate Directional Movement
    df['up_move'] = df['high'] - df['high'].shift()
    df['down_move'] = df['low'].shift() - df['low']

    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
        df['up_move'],
        0
    )
    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
        df['down_move'],
        0
    )

    # Calculate ADX
    df['+di'] = 100 * (df['plus_dm'].ewm(span=period).mean() / 
                       df['tr'].ewm(span=period).mean())
    df['-di'] = 100 * (df['minus_dm'].ewm(span=period).mean() / 
                       df['tr'].ewm(span=period).mean())
    df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
    df['adx'] = df['dx'].ewm(span=period).mean()

    return df[['adx', '+di', '-di']]

def calculate_ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud indicators
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        Dictionary containing Ichimoku components
    """
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df['high'].rolling(window=9).max()
    period9_low = df['low'].rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df['high'].rolling(window=26).max()
    period26_low = df['low'].rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = df['high'].rolling(window=52).max()
    period52_low = df['low'].rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Lagging Span): Close shifted back 26 periods
    chikou_span = df['close'].shift(-26)

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range
    
    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation
    
    Returns:
        pd.Series: ATR values
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()

def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series
    
    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation
    
    Returns:
        float: Hurst exponent
    """
    lags = range(2, max_lag)
    # Add small epsilon to avoid log(0)
    tau = [max(1e-8, np.sqrt(np.std(np.subtract(price_series[lag:].values, price_series[:-lag].values)))) for lag in lags]

    # Return the Hurst exponent from linear fit
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0] # Hurst exponent is the slope
    except (ValueError, RuntimeWarning):
        # Return 0.5 (random walk) if calculation fails
        return 0.5
