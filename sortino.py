import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os, ssl
import requests  # Add requests import

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',  # Do Not Track Request Header
        'Connection': 'close'
    }

def get_cedears():
    url = "https://www.comafi.com.ar/custodiaglobal/json/apps/getproducts.aspx?ts=125.3.8.12.23"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        # Assuming the JSON structure is like {"products": [{"name": "...", ...}, ...]}
        if 'products' in data and isinstance(data['products'], list):
            cedears = [product.get('name') for product in data['products'] if product.get('name')]
            return cedears
        else:
            print("Error: 'products' key not found or not a list in JSON response.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []
    except ValueError as e: # Includes JSONDecodeError
        print(f"Error parsing JSON: {e}")
        return []

def analyze_sector_with_sortino(sector_name):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]

    tickers = sp500_table[sp500_table["GICS Sector"] == sector_name]["Symbol"].tolist()
    sector_tickers = ['BRK-B' if ticker in ['BRK.B', 'BRK/B'] else ticker for ticker in tickers]

    today = datetime.today()
    two_years_ago = today - timedelta(days=2*365)
    five_years_ago = today - timedelta(days=5*365)

    two_years_ago_str = two_years_ago.strftime('%Y-%m-%d')
    five_years_ago_str = five_years_ago.strftime('%Y-%m-%d')

    data = yf.download(sector_tickers, start=five_years_ago_str, end=today.strftime('%Y-%m-%d'))["Close"]
    sp500_data = yf.download("^GSPC", start=five_years_ago_str, end=today.strftime('%Y-%m-%d'))["Close"]
    sp500_returns = sp500_data.pct_change()

    risk_free_rate = 0.02

    # Cálculo de Sortino Ratio del S&P 500 ajustado
    downside_sp500 = sp500_returns[sp500_returns < 0].std() * np.sqrt(252)  # Anualizamos la desviación hacia abajo
    sortino_sp500_2yr = ((sp500_returns[sp500_returns.index >= two_years_ago_str].mean() * 252 - risk_free_rate) 
                         / downside_sp500)
    sortino_sp500_5yr = ((sp500_returns.mean() * 252 - risk_free_rate) / downside_sp500)

    returns = data.pct_change()
    returns_2yr = returns[returns.index >= two_years_ago_str]
    returns_5yr = returns

    sortino_ratios = {}
    for ticker in sector_tickers:
        # Sortino Ratio para 2 años
        downside_2yr = returns_2yr[ticker][returns_2yr[ticker] < 0].std() * np.sqrt(252)
        sortino_2yr_ticker = (returns_2yr[ticker].mean() * 252 - risk_free_rate) / downside_2yr if downside_2yr != 0 else np.nan

        # Sortino Ratio para 5 años
        downside_5yr = returns_5yr[ticker][returns_5yr[ticker] < 0].std() * np.sqrt(252)
        sortino_5yr_ticker = (returns_5yr[ticker].mean() * 252 - risk_free_rate) / downside_5yr if downside_5yr != 0 else np.nan

        # Get last price and date
        last_valid_date = data[ticker].last_valid_index()
        last_price = data.loc[last_valid_date, ticker] if pd.notna(last_valid_date) else np.nan
        last_date_str = last_valid_date.strftime('%Y-%m-%d') if pd.notna(last_valid_date) else 'N/A'

        sortino_ratios[ticker] = (sortino_2yr_ticker, sortino_5yr_ticker, last_price, last_date_str)

    cedears = get_cedears()
    # Update DataFrame creation to include new columns
    sortino_df = pd.DataFrame.from_dict(sortino_ratios, orient='index', columns=["2 Years", "5 Years", "Last Price", "Last Date"])
    sortino_df.index.name = 'Ticker' # Optional: Name the index

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 7), dpi=120)

    for ticker in sortino_df.index:
        
        if ticker in cedears:
            ax.scatter(sortino_df.loc[ticker, "2 Years"], sortino_df.loc[ticker, "5 Years"], s=200, color="magenta", alpha=0.7, label=f"{ticker} (CEDEAR)")
            if (sortino_df.loc[ticker, "2 Years"] > sortino_sp500_2yr).all() and (sortino_df.loc[ticker, "5 Years"] > sortino_sp500_5yr).all():
                cedears_selection.append(ticker)
        else:
            ax.scatter(sortino_df.loc[ticker, "2 Years"], sortino_df.loc[ticker, "5 Years"], s=200, color="cyan", alpha=0.3, label=f"{ticker} (Stock)")

    ax.scatter(sortino_sp500_2yr, sortino_sp500_5yr, color="red", alpha=0.5, s=550, label="S&P 500")
    ax.text(sortino_sp500_2yr, sortino_sp500_5yr, "SP500", color="white", fontsize=8, ha="center", va="center")

    for ticker in sortino_df.index:
        ax.text(sortino_df.loc[ticker, "2 Years"], sortino_df.loc[ticker, "5 Years"], ticker, color="white", fontsize=6, alpha=0.99, ha="center", va="center")

    ax.axhline(sortino_sp500_5yr.item(), color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(sortino_sp500_2yr.item(), color="gray", linestyle="--", linewidth=0.8)

    # Keep a copy before dropping NaNs for the CSV output, but drop for plotting/regression
    sortino_df_full = sortino_df.copy() 
    sortino_df_cleaned = sortino_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["2 Years", "5 Years"])

    # Use cleaned data for plotting and regression
    x = sortino_df_cleaned["2 Years"]
    y = sortino_df_cleaned["5 Years"]
    
    # Calculate sorted values for regression AFTER defining x and y from cleaned data
    sorted_indices = np.argsort(x)
    x_sorted = x.iloc[sorted_indices]
    y_sorted = y.iloc[sorted_indices]

    # Check if there's enough data for regression after cleaning
    if len(x_sorted) > 1: 
        slope, intercept = np.polyfit(x_sorted, y_sorted, 1)
        regression_line = slope * x_sorted + intercept
        ax.plot(x_sorted, regression_line, color="red", linestyle="--", linewidth=1.5, label="Línea de Regresión", alpha=0.5)
    else:
        print(f"Warning: Not enough data points ({len(x_sorted)}) for regression line in sector {sector_name}")


    ax.set_xlabel("Sortino Ratio (2 Years)", color="white")
    ax.set_ylabel("Sortino Ratio (5 Years)", color="white")
    ax.set_title(f"{sector_name} Sector Risk Quadrant (Sortino Ratio)", color="white")
    os.makedirs("./charts", exist_ok=True)  # Ensure the directory exists
    plt.savefig(os.path.join("./charts", f"{sector_name}.png"), dpi=300)
    plt.close(fig) # Close the figure to free memory
    # plt.show()
    return sortino_df_full # Return the full dataframe (including NaNs) for CSV export


if __name__ == "__main__":

    sectors = ['Industrials','Health Care','Information Technology','Utilities','Financials','Materials','Consumer Discretionary','Real Estate','Communication Services','Consumer Staples','Energy']
    cedears_selection = []
    all_sector_dfs = [] # List to store dataframes from each sector

    print("Analyzing sectors...")
    for i, sector in enumerate(sectors):
        print(f"Processing sector {i+1}/{len(sectors)}: {sector}")
        try:
            sector_df = analyze_sector_with_sortino(sector)
            if not sector_df.empty:
                sector_df['Sector'] = sector # Add sector column
                all_sector_dfs.append(sector_df)
            else:
                print(f"Warning: No valid Sortino data generated for sector {sector}")
        except Exception as e:
            print(f"Error processing sector {sector}: {e}")

    # Combine all sector dataframes
    if all_sector_dfs:
        combined_sortino_df = pd.concat(all_sector_dfs)
        # Save the combined dataframe to CSV
        output_csv_path = "all_sortino_ratios.csv"
        combined_sortino_df.to_csv(output_csv_path)
        print(f"Combined Sortino ratios saved to {output_csv_path}")
    else:
        print("No dataframes to combine. Output CSV not created.")


    # Save the cedears selection as before
    output_txt_path = "cedears_selection.txt"
    with open(output_txt_path, "w") as file:
        if cedears_selection:
            file.write(", ".join(cedears_selection))
            file.write('\nhttps://finviz.com/screener.ashx?v=340&t=' + ','.join(cedears_selection))
        else:
            file.write("No CEDEARs met the selection criteria.")
    print(f"CEDEARs selection saved to {output_txt_path}")
