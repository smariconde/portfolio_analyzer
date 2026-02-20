import pandas as pd
from yahooquery import Ticker
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
import os, ssl
import requests
from io import StringIO
import time
import pickle
from pathlib import Path
import textwrap

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',  # Do Not Track Request Header
        'Connection': 'close'
    }

# Create cache directory
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(ticker, start_date, end_date):
    """Generate cache file path for a ticker"""
    return CACHE_DIR / f"{ticker}_{start_date}_{end_date}.pkl"

def load_from_cache(ticker, start_date, end_date, max_age_days=1):
    """Load data from cache if available and recent"""
    cache_path = get_cache_path(ticker, start_date, end_date)
    if cache_path.exists():
        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age.days <= max_age_days:
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache for {ticker}: {e}")
    return None

def save_to_cache(ticker, data, start_date, end_date):
    """Save data to cache"""
    try:
        cache_path = get_cache_path(ticker, start_date, end_date)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving cache for {ticker}: {e}")

def download_ticker_with_cache(ticker, start, end, max_retries=3):
    """Download single ticker with cache support using yahooquery"""
    # Try to load from cache first
    cached_data = load_from_cache(ticker, start, end)
    if cached_data is not None:
        print(f"  Loaded {ticker} from cache")
        return cached_data

    # Download from yahooquery
    for attempt in range(max_retries):
        try:
            ticker_obj = Ticker(ticker, asynchronous=False)
            data = ticker_obj.history(start=start, end=end, interval='1d')

            # yahooquery returns a dict-like response, check for errors
            if isinstance(data, dict) and ticker in data:
                error_msg = data[ticker]
                print(f"  Error for {ticker}: {error_msg}")
                return None

            # Check if data is valid
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Reset index to have dates as column if needed
                if data.index.name == 'date' or 'date' in data.index.names:
                    # yahooquery returns MultiIndex (symbol, date), we want just dates
                    if isinstance(data.index, pd.MultiIndex):
                        data = data.droplevel(0)  # Remove symbol level, keep date

                save_to_cache(ticker, data, start, end)
                print(f"  Downloaded {ticker}")
                return data
            else:
                print(f"  No data for {ticker}")
                return None

        except Exception as e:
            error_str = str(e)
            if "Rate" in error_str or "429" in error_str or "limit" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"  Rate limited on {ticker}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed {ticker} after {max_retries} retries")
                    return None
            else:
                print(f"  Error downloading {ticker}: {e}")
                return None
    return None

def download_tickers_in_batches(tickers, start, end, batch_size=10, delay_between_batches=2):
    """Download multiple tickers in batches using yahooquery's asynchronous mode"""
    all_data = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches}: {', '.join(batch)}")

        # First check cache for each ticker in batch
        uncached_tickers = []
        for ticker in batch:
            cached_data = load_from_cache(ticker, start, end)
            if cached_data is not None:
                print(f"    Loaded {ticker} from cache")
                all_data[ticker] = cached_data
            else:
                uncached_tickers.append(ticker)

        # Track if we actually downloaded anything new
        downloaded_new_data = False

        # Download uncached tickers using yahooquery's async mode
        if uncached_tickers:
            downloaded_new_data = True
            try:
                # Use yahooquery's asynchronous mode for batch download
                ticker_string = ' '.join(uncached_tickers)
                ticker_obj = Ticker(ticker_string, asynchronous=True)
                batch_data = ticker_obj.history(start=start, end=end, interval='1d')

                if isinstance(batch_data, pd.DataFrame) and not batch_data.empty:
                    # yahooquery returns MultiIndex DataFrame with (symbol, date)
                    for ticker in uncached_tickers:
                        try:
                            if ticker in batch_data.index.get_level_values(0):
                                ticker_data = batch_data.xs(ticker, level=0)
                                save_to_cache(ticker, ticker_data, start, end)
                                all_data[ticker] = ticker_data
                                print(f"    Downloaded {ticker}")
                            else:
                                print(f"    No data for {ticker}")
                        except Exception as e:
                            print(f"    Error extracting {ticker}: {e}")
                else:
                    print(f"    No data returned for batch")

            except Exception as e:
                print(f"    Batch download failed: {e}, falling back to individual downloads")
                # Fallback to individual downloads
                for ticker in uncached_tickers:
                    data = download_ticker_with_cache(ticker, start, end)
                    if data is not None and not data.empty:
                        all_data[ticker] = data
                    time.sleep(1)  # Small delay between individual downloads

        # Only delay if we downloaded new data AND there are more batches
        if i + batch_size < len(tickers):
            if downloaded_new_data:
                print(f"  Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)
            else:
                print(f"  All from cache, skipping delay")

    return all_data

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

def analyze_sector_with_sortino(sector_name, sp500_data, sp500_returns, sortino_sp500_2yr, sortino_sp500_5yr, two_years_ago_dt):
    """Analyze a sector with Sortino ratios using pre-downloaded S&P 500 data"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    sp500_table = tables[0]

    tickers = sp500_table[sp500_table["GICS Sector"] == sector_name]["Symbol"].tolist()
    sector_tickers = ['BRK-B' if ticker in ['BRK.B', 'BRK/B'] else ticker for ticker in tickers]

    today = datetime.today()
    five_years_ago = today - timedelta(days=5*365)
    five_years_ago_str = five_years_ago.strftime('%Y-%m-%d')

    print(f"  Downloading {len(sector_tickers)} tickers for {sector_name}...")
    # Download data in batches to avoid rate limiting
    ticker_data_dict = download_tickers_in_batches(sector_tickers, start=five_years_ago_str, end=today.strftime('%Y-%m-%d'))

    # Convert to DataFrame with Close prices
    data = pd.DataFrame()
    for ticker, ticker_df in ticker_data_dict.items():
        if isinstance(ticker_df, pd.Series):
            data[ticker] = ticker_df
        elif isinstance(ticker_df, pd.DataFrame):
            # yahooquery returns columns: open, high, low, close, volume, adjclose (lowercase)
            if 'close' in ticker_df.columns:
                data[ticker] = ticker_df['close']
            elif 'Close' in ticker_df.columns:
                data[ticker] = ticker_df['Close']
            elif 'adjclose' in ticker_df.columns:
                data[ticker] = ticker_df['adjclose']
            elif len(ticker_df.columns) == 1:
                data[ticker] = ticker_df.iloc[:, 0]
            else:
                # Default to first column if we can't find close
                print(f"    Warning: Couldn't find close price for {ticker}, using first column")
                print(f"    Available columns: {ticker_df.columns.tolist()}")
                data[ticker] = ticker_df.iloc[:, 0]

    if data.empty:
        print(f"  Warning: No data downloaded for sector {sector_name}")
        return pd.DataFrame(), []

    print(f"  Successfully downloaded {len(data.columns)}/{len(sector_tickers)} tickers")

    # Ensure index is DatetimeIndex for proper comparison
    data.index = pd.DatetimeIndex(data.index)

    risk_free_rate = 0.02

    returns = data.pct_change()
    # Use the two_years_ago_dt parameter directly (already a Timestamp)
    returns_2yr = returns[returns.index >= two_years_ago_dt]
    returns_5yr = returns

    sortino_ratios = {}
    # Only calculate for tickers that were successfully downloaded
    for ticker in data.columns:
        try:
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
        except Exception as e:
            print(f"  Error calculating Sortino for {ticker}: {e}")
            continue

    cedears = get_cedears()
    # Update DataFrame creation to include new columns
    sortino_df = pd.DataFrame.from_dict(sortino_ratios, orient='index', columns=["2 Years", "5 Years", "Last Price", "Last Date"])
    sortino_df.index.name = 'Ticker' # Optional: Name the index

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

    # Initialize cedears_selection for the current sector analysis
    cedears_selection_sector = []

    # Separate plotting by type to create clean legend
    cedear_tickers = []
    stock_tickers = []

    for ticker in sortino_df.index:
        # Check if the ticker or its alternative form ('BRK/B' for 'BRK-B') is in cedears
        is_cedear = ticker in cedears or (ticker == 'BRK-B' and 'BRK/B' in cedears)

        if is_cedear:
            cedear_tickers.append(ticker)
            ax.scatter(sortino_df.loc[ticker, "2 Years"], sortino_df.loc[ticker, "5 Years"],
                      s=200, color="magenta", alpha=0.7)
            # Check selection criteria - FIXED: removed .all() since these are scalar comparisons
            if sortino_df.loc[ticker, "2 Years"] > sortino_sp500_2yr and sortino_df.loc[ticker, "5 Years"] > sortino_sp500_5yr:
                cedears_selection_sector.append(ticker)
        else:
            stock_tickers.append(ticker)
            ax.scatter(sortino_df.loc[ticker, "2 Years"], sortino_df.loc[ticker, "5 Years"],
                      s=200, color="cyan", alpha=0.3)

    ax.scatter(sortino_sp500_2yr, sortino_sp500_5yr, color="red", alpha=0.5, s=550, label="S&P 500")
    ax.text(sortino_sp500_2yr, sortino_sp500_5yr, "SP500", color="white", fontsize=8, ha="center", va="center")

    for ticker in sortino_df.index:
        ax.text(sortino_df.loc[ticker, "2 Years"], sortino_df.loc[ticker, "5 Years"], ticker, color="white", fontsize=6, alpha=0.99, ha="center", va="center")

    # Reference lines for S&P 500
    sp500_2yr_val = float(sortino_sp500_2yr.iloc[0]) if isinstance(sortino_sp500_2yr, pd.Series) else sortino_sp500_2yr
    sp500_5yr_val = float(sortino_sp500_5yr.iloc[0]) if isinstance(sortino_sp500_5yr, pd.Series) else sortino_sp500_5yr

    ax.axhline(sp500_5yr_val, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(sp500_2yr_val, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

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
        ax.plot(x_sorted, regression_line, color="yellow", linestyle="--", linewidth=1.5, alpha=0.4)

    # Add quadrant labels
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()

    # Top-right quadrant (best performers)
    ax.text(ax_xlim[1] * 0.95, ax_ylim[1] * 0.95, 'Best Performers\n(Beat S&P 500)',
            ha='right', va='top', fontsize=9, color='lime', alpha=0.7, weight='bold')

    # Bottom-left quadrant (worst performers)
    ax.text(ax_xlim[0] * 1.05, ax_ylim[0] * 1.05, 'Underperformers',
            ha='left', va='bottom', fontsize=9, color='red', alpha=0.7, weight='bold')

    # Create clean legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor='magenta', alpha=0.7, label=f'CEDEARs ({len(cedear_tickers)})'),
        Patch(facecolor='cyan', alpha=0.3, label=f'US Stocks ({len(stock_tickers)})'),
        Patch(facecolor='red', alpha=0.5, label='S&P 500 Index'),
    ]
    if len(x_sorted) > 1:
        legend_elements.append(Line2D([0], [0], color='yellow', linestyle='--', alpha=0.4, label='Trend Line'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    # Title with date and statistics
    analysis_date = datetime.today().strftime('%Y-%m-%d')
    title = f"{sector_name} Sector - Sortino Ratio Analysis"
    subtitle = f"Generated: {analysis_date} | Total: {len(sortino_df)} tickers | CEDEARs Selected: {len(cedears_selection_sector)}"

    ax.set_xlabel("Sortino Ratio (2 Years)", color="white", fontsize=12)
    ax.set_ylabel("Sortino Ratio (5 Years)", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, weight='bold', pad=20)

    # Add subtitle with better spacing
    fig.text(0.5, 0.96, subtitle, ha='center', va='top',
            color='lightgray', fontsize=10, style='italic')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs("./charts", exist_ok=True)  # Ensure the directory exists
    # Save with higher DPI for better quality in PDF
    plt.savefig(os.path.join("./charts", f"{sector_name}.png"), dpi=450, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    # plt.show()
    # Return the full dataframe and the sector-specific selection
    return sortino_df_full, cedears_selection_sector


def generate_analysis_pdf(all_sector_dfs, cedears_selection_global, sectors, analysis_date):
    """Generate a comprehensive PDF report with all analysis results"""
    pdf_filename = f"Sortino_Analysis_Report_{analysis_date}.pdf"

    with PdfPages(pdf_filename) as pdf:
        # Page 1: Cover page with summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('#1a1a1a')
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Title
        ax.text(0.5, 0.85, 'S&P 500 Sortino Ratio Analysis', ha='center', va='top',
                fontsize=24, weight='bold', color='white', transform=ax.transAxes)
        ax.text(0.5, 0.80, 'CEDEAR Investment Opportunities', ha='center', va='top',
                fontsize=16, color='lightblue', transform=ax.transAxes, style='italic')

        # Date
        ax.text(0.5, 0.72, f'Analysis Date: {analysis_date}', ha='center', va='top',
                fontsize=14, color='lightgray', transform=ax.transAxes)

        # Summary statistics
        combined_df = pd.concat(all_sector_dfs) if all_sector_dfs else pd.DataFrame()
        total_tickers = len(combined_df)
        total_sectors = len(sectors)
        unique_cedears = sorted(list(set(cedears_selection_global)))

        # Format CEDEARs list with automatic line breaks
        if unique_cedears:
            cedears_str = ', '.join(unique_cedears)
            # Wrap text to 70 characters per line with proper indentation
            wrapped_cedears = textwrap.fill(cedears_str, width=70,
                                           initial_indent='        ',
                                           subsequent_indent='        ')
        else:
            wrapped_cedears = '        None'

        summary_text = f"""
        ANALYSIS SUMMARY
        {'='*50}

        Total Sectors Analyzed: {total_sectors}
        Total Tickers Evaluated: {total_tickers}

        CEDEARs Selected (Beat S&P 500): {len(unique_cedears)}

        Selected CEDEARs:
{wrapped_cedears}

        {'='*50}

        METHODOLOGY
        • Sortino Ratio: Risk-adjusted return metric
        • Time Periods: 2 years & 5 years
        • Benchmark: S&P 500 Index
        • Selection Criteria: Beat S&P 500 in both periods
        • Risk-Free Rate: 2.0%

        {'='*50}
        """

        ax.text(0.1, 0.65, summary_text, ha='left', va='top',
                fontsize=10, color='white', transform=ax.transAxes,
                family='monospace', bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

        # Footer
        ax.text(0.5, 0.05, 'Generated with Python • yahooquery • matplotlib', ha='center', va='bottom',
                fontsize=8, color='gray', transform=ax.transAxes, style='italic')

        pdf.savefig(fig, facecolor='#1a1a1a')
        plt.close(fig)

        # Page 2: Selected CEDEARs Table
        if unique_cedears and not combined_df.empty:
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor('#1a1a1a')
            ax = fig.add_subplot(111)
            ax.axis('off')

            ax.text(0.5, 0.95, 'Selected CEDEARs - Detailed Analysis', ha='center', va='top',
                    fontsize=18, weight='bold', color='white', transform=ax.transAxes)

            # Filter data for selected CEDEARs
            cedears_data = combined_df[combined_df.index.isin(unique_cedears)].copy()
            cedears_data = cedears_data.sort_values('2 Years', ascending=False)

            # Create table
            table_data = []
            table_data.append(['Ticker', 'Sector', '2Y Sortino', '5Y Sortino', 'Last Price', 'Last Date'])

            for ticker in cedears_data.index:
                row_data = cedears_data.loc[ticker]
                table_data.append([
                    ticker,
                    row_data['Sector'][:15],  # Truncate long sector names
                    f"{row_data['2 Years']:.2f}" if pd.notna(row_data['2 Years']) else 'N/A',
                    f"{row_data['5 Years']:.2f}" if pd.notna(row_data['5 Years']) else 'N/A',
                    f"${row_data['Last Price']:.2f}" if pd.notna(row_data['Last Price']) else 'N/A',
                    str(row_data['Last Date'])[:10]
                ])

            # Create table plot
            table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                           bbox=[0.05, 0.1, 0.9, 0.8])

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)

            # Style header row
            for i in range(6):
                cell = table[(0, i)]
                cell.set_facecolor('#3a3a3a')
                cell.set_text_props(weight='bold', color='white')

            # Style data rows
            for i in range(1, len(table_data)):
                for j in range(6):
                    cell = table[(i, j)]
                    cell.set_facecolor('#2a2a2a' if i % 2 == 0 else '#252525')
                    cell.set_text_props(color='white')

            pdf.savefig(fig, facecolor='#1a1a1a')
            plt.close(fig)

        # Pages 3+: Include existing sector charts with high quality
        print("  Adding sector charts to PDF...")
        from PIL import Image
        for sector in sectors:
            chart_path = os.path.join("./charts", f"{sector}.png")
            if os.path.exists(chart_path):
                # Read the high-res chart image with PIL for better quality
                img = Image.open(chart_path)

                # Create figure matching the image aspect ratio
                img_width, img_height = img.size
                fig_width = 11  # inches
                fig_height = fig_width * (img_height / img_width)

                fig = plt.figure(figsize=(fig_width, fig_height))
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                plt.tight_layout(pad=0)

                # Save with high DPI to maintain quality
                pdf.savefig(fig, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

        # Last Page: Statistics Summary by Sector
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('#1a1a1a')
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.95, 'Sector Statistics Summary', ha='center', va='top',
                fontsize=18, weight='bold', color='white', transform=ax.transAxes)

        # Create sector summary table
        sector_stats = []
        sector_stats.append(['Sector', 'Tickers', 'Avg 2Y', 'Avg 5Y', 'Best 2Y', 'Best 5Y'])

        for sector in sectors:
            sector_data = combined_df[combined_df['Sector'] == sector]
            if not sector_data.empty:
                sector_stats.append([
                    sector[:20],
                    str(len(sector_data)),
                    f"{sector_data['2 Years'].mean():.2f}" if sector_data['2 Years'].notna().any() else 'N/A',
                    f"{sector_data['5 Years'].mean():.2f}" if sector_data['5 Years'].notna().any() else 'N/A',
                    f"{sector_data['2 Years'].max():.2f}" if sector_data['2 Years'].notna().any() else 'N/A',
                    f"{sector_data['5 Years'].max():.2f}" if sector_data['5 Years'].notna().any() else 'N/A',
                ])

        table = ax.table(cellText=sector_stats, cellLoc='center', loc='center',
                       bbox=[0.05, 0.1, 0.9, 0.8])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)

        # Style header
        for i in range(6):
            cell = table[(0, i)]
            cell.set_facecolor('#3a3a3a')
            cell.set_text_props(weight='bold', color='white')

        # Style data rows
        for i in range(1, len(sector_stats)):
            for j in range(6):
                cell = table[(i, j)]
                cell.set_facecolor('#2a2a2a' if i % 2 == 0 else '#252525')
                cell.set_text_props(color='white')

        pdf.savefig(fig, facecolor='#1a1a1a')
        plt.close(fig)

        # Set PDF metadata
        d = pdf.infodict()
        d['Title'] = f'S&P 500 Sortino Analysis Report - {analysis_date}'
        d['Author'] = 'Sortino Analysis Script'
        d['Subject'] = 'CEDEAR Investment Analysis'
        d['Keywords'] = 'Sortino Ratio, S&P 500, CEDEARs, Investment Analysis'
        d['CreationDate'] = datetime.today()

    print(f"\n{'='*60}")
    print(f"PDF Report generated: {pdf_filename}")
    print(f"{'='*60}")
    return pdf_filename


if __name__ == "__main__":

    sectors = ['Industrials','Health Care','Information Technology','Utilities','Financials','Materials','Consumer Discretionary','Real Estate','Communication Services','Consumer Staples','Energy']
    cedears_selection_global = []
    all_sector_dfs = []

    # Calculate date ranges once
    today = datetime.today()
    two_years_ago = today - timedelta(days=2*365)
    five_years_ago = today - timedelta(days=5*365)
    two_years_ago_str = two_years_ago.strftime('%Y-%m-%d')
    five_years_ago_str = five_years_ago.strftime('%Y-%m-%d')

    # Download S&P 500 data once at the beginning
    print("Downloading S&P 500 data...")
    sp500_data = download_ticker_with_cache("^GSPC", start=five_years_ago_str, end=today.strftime('%Y-%m-%d'))

    if sp500_data is None or sp500_data.empty:
        print("ERROR: Failed to download S&P 500 data. Cannot proceed.")
        exit(1)

    # Extract Close prices from S&P 500 data
    if isinstance(sp500_data, pd.Series):
        sp500_close = sp500_data
    elif isinstance(sp500_data, pd.DataFrame):
        # yahooquery returns lowercase column names
        if 'close' in sp500_data.columns:
            sp500_close = sp500_data['close']
        elif 'Close' in sp500_data.columns:
            sp500_close = sp500_data['Close']
        elif 'adjclose' in sp500_data.columns:
            sp500_close = sp500_data['adjclose']
        else:
            sp500_close = sp500_data.iloc[:, 0]
    else:
        sp500_close = sp500_data.iloc[:, 0]

    # Ensure index is DatetimeIndex for proper comparison
    sp500_close.index = pd.DatetimeIndex(sp500_close.index)

    sp500_returns = sp500_close.pct_change()

    # Calculate S&P 500 Sortino ratios once
    risk_free_rate = 0.02
    downside_sp500 = sp500_returns[sp500_returns < 0].std() * np.sqrt(252)

    # Convert two_years_ago to Timestamp for comparison with index
    two_years_ago_dt = pd.Timestamp(two_years_ago)

    sortino_sp500_2yr = ((sp500_returns[sp500_returns.index >= two_years_ago_dt].mean() * 252 - risk_free_rate) / downside_sp500)
    sortino_sp500_5yr = ((sp500_returns.mean() * 252 - risk_free_rate) / downside_sp500)

    print(f"S&P 500 Sortino Ratio (2yr): {sortino_sp500_2yr:.3f}")
    print(f"S&P 500 Sortino Ratio (5yr): {sortino_sp500_5yr:.3f}")
    print("\nAnalyzing sectors...")

    for i, sector in enumerate(sectors):
        print(f"\n{'='*60}")
        print(f"Processing sector {i+1}/{len(sectors)}: {sector}")
        print(f"{'='*60}")
        try:
            # Pass pre-downloaded S&P 500 data to avoid redundant downloads
            sector_df, sector_cedears = analyze_sector_with_sortino(
                sector,
                sp500_close,
                sp500_returns,
                sortino_sp500_2yr,
                sortino_sp500_5yr,
                two_years_ago_dt
            )

            if not sector_df.empty:
                sector_df['Sector'] = sector
                all_sector_dfs.append(sector_df)
                cedears_selection_global.extend(sector_cedears)

                # Save incremental progress after each successful sector
                if all_sector_dfs:
                    combined_sortino_df = pd.concat(all_sector_dfs)
                    output_csv_path = "all_sortino_ratios.csv"
                    combined_sortino_df.to_csv(output_csv_path)
                    print(f"  Progress saved to {output_csv_path} ({len(all_sector_dfs)} sectors completed)")
            else:
                print(f"  Warning: No valid Sortino data generated for sector {sector}")
        except Exception as e:
            print(f"  ERROR processing sector {sector}: {e}")
            import traceback
            traceback.print_exc()
            # Continue with next sector even if this one fails

        # Shorter delay between sectors since we're downloading in small batches now
        if i < len(sectors) - 1:
            delay = 5
            print(f"\nWaiting {delay} seconds before next sector...")
            time.sleep(delay)

    # Final save
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

    if all_sector_dfs:
        combined_sortino_df = pd.concat(all_sector_dfs)
        output_csv_path = "all_sortino_ratios.csv"
        combined_sortino_df.to_csv(output_csv_path)
        print(f"Final results saved to {output_csv_path}")
        print(f"Total tickers analyzed: {len(combined_sortino_df)}")
        print(f"Total sectors completed: {len(all_sector_dfs)}/{len(sectors)}")
    else:
        print("ERROR: No dataframes to combine. Output CSV not created.")

    # Save the global cedears selection
    output_txt_path = "cedears_selection.txt"
    with open(output_txt_path, "w") as file:
        if cedears_selection_global:
            unique_cedears = sorted(list(set(cedears_selection_global)))
            file.write(", ".join(unique_cedears))
            file.write('\nhttps://finviz.com/screener.ashx?v=340&t=' + ','.join(unique_cedears))
            print(f"CEDEARs selection saved to {output_txt_path}")
            print(f"Total CEDEARs selected: {len(unique_cedears)}")
        else:
            file.write("No CEDEARs met the selection criteria.")
            print(f"No CEDEARs met the selection criteria.")

    # Generate comprehensive PDF report
    if all_sector_dfs:
        print(f"\n{'='*60}")
        print("Generating PDF Report...")
        print(f"{'='*60}")
        analysis_date_str = datetime.today().strftime('%Y-%m-%d')
        pdf_filename = generate_analysis_pdf(all_sector_dfs, cedears_selection_global, sectors, analysis_date_str)
        print(f"\n✓ All outputs generated successfully!")
        print(f"  - CSV: {output_csv_path}")
        print(f"  - TXT: {output_txt_path}")
        print(f"  - PDF: {pdf_filename}")
        print(f"  - Charts: ./charts/")
    else:
        print("\nSkipping PDF generation (no data available)")
