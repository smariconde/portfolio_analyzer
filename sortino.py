import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os, ssl

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',  # Do Not Track Request Header
        'Connection': 'close'
    }

def get_cedears():
    url = "https://www.comafi.com.ar/custodiaglobal/2483-Programas-Cedear.cedearnote.note.aspx#shares"
    tables = pd.read_html(url)
    cedear_table = tables[0]
    cedears = cedear_table.iloc[:, 5].tolist()
    return cedears

def analyze_sector_with_sortino(sector_name):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]

    tickers = sp500_table[sp500_table["GICS Sector"] == sector_name]["Symbol"].tolist()
    sector_tickers = ['BRK-B' if ticker == 'BRK.B' else ticker for ticker in tickers]

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

        sortino_ratios[ticker] = (sortino_2yr_ticker, sortino_5yr_ticker)

    cedears = get_cedears()
    sortino_df = pd.DataFrame(sortino_ratios, index=["2 Years", "5 Years"]).T
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

    sortino_df = sortino_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["2 Years", "5 Years"])

    x = sortino_df["2 Years"]
    y = sortino_df["5 Years"]
    sorted_indices = np.argsort(x)
    x_sorted = x.iloc[sorted_indices]
    y_sorted = y.iloc[sorted_indices]

    slope, intercept = np.polyfit(x_sorted, y_sorted, 1)
    regression_line = slope * x_sorted + intercept

    ax.plot(x_sorted, regression_line, color="red", linestyle="--", linewidth=1.5, label="Línea de Regresión", alpha=0.5)

    ax.set_xlabel("Sortino Ratio (2 Years)", color="white")
    ax.set_ylabel("Sortino Ratio (5 Years)", color="white")
    ax.set_title(f"{sector_name} Sector Risk Quadrant (Sortino Ratio)", color="white")
    os.makedirs("./charts", exist_ok=True)  # Ensure the directory exists
    plt.savefig(os.path.join("./charts", f"{sector_name}.png"), dpi=300)
    # plt.show()


if __name__ == "__main__":

    sectors = ['Industrials','Health Care','Information Technology','Utilities','Financials','Materials','Consumer Discretionary','Real Estate','Communication Services','Consumer Staples','Energy']
    cedears_selection = []
    for sector in sectors:
        analyze_sector_with_sortino(sector)

    with open("cedears_selection.txt", "w") as file:
        file.write(", ".join(cedears_selection))
        file.write('\n https://finviz.com/screener.ashx?v=340&t=' + ','.join(cedears_selection))
