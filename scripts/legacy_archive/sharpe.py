import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_cedears():
    # Obtención de la lista de CEDEARs desde el sitio web de Comafi
    url = "https://www.comafi.com.ar/custodiaglobal/2483-Programas-Cedear.cedearnote.note.aspx#shares"
    tables = pd.read_html(url)
    cedear_table = tables[0]
    
    # Filtramos los tickers de los CEDEARs (ajustar el índice de las columnas según el formato de la tabla)
    cedears = cedear_table.iloc[:, 5].tolist()  # La columna 6 (índice 5) contiene los tickers
    return cedears

def analyze_sector_with_sharpe(sector_name):
    # 1. Scraping de Wikipedia para obtener los tickers del S&P 500 y sus sectores
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]  # La primera tabla en la página suele contener la lista de empresas

    # Filtrar empresas según el sector proporcionado
    sector_tickers = sp500_table[sp500_table["GICS Sector"] == sector_name]["Symbol"].tolist()

    # Obtener la fecha actual
    today = datetime.today()

    # Calcular las fechas de 2 y 5 años atrás
    two_years_ago = today - timedelta(days=2*365)  # Aproximación de 2 años
    five_years_ago = today - timedelta(days=5*365)  # Aproximación de 5 años

    # Convertir a formato de fecha compatible con pandas
    two_years_ago_str = two_years_ago.strftime('%Y-%m-%d')
    five_years_ago_str = five_years_ago.strftime('%Y-%m-%d')

    # Descargar datos de precios de cierre ajustados para el sector especificado
    data = yf.download(sector_tickers, start=five_years_ago_str, end=today.strftime('%Y-%m-%d'))["Adj Close"]

    # Descargar datos del S&P 500 para el punto de referencia
    sp500_data = yf.download("^GSPC", start=five_years_ago_str, end=today.strftime('%Y-%m-%d'))["Adj Close"]
    sp500_returns = sp500_data.pct_change()

    risk_free_rate = 0.02  # Tasa libre de riesgo anual (ajusta según corresponda)

    # Calcular Sharpe Ratio de referencia para el S&P 500 a 2 años y a 5 años
    sp500_sharpe_ratio_2yr = (sp500_returns[sp500_returns.index >= two_years_ago_str].mean() * 252 - risk_free_rate) / (sp500_returns[sp500_returns.index >= two_years_ago_str].std() * np.sqrt(252))
    sp500_sharpe_ratio_5yr = (sp500_returns.mean() * 252 - risk_free_rate) / (sp500_returns.std() * np.sqrt(252))

    # 2. Calcular retornos diarios
    returns = data.pct_change()

    # 3. Dividir los datos en ventanas de 2 años y 5 años
    returns_2yr = returns[returns.index >= two_years_ago_str]
    returns_5yr = returns

    # Cálculo del Sharpe Ratio para 2 y 5 años
    sharpe_ratios = {}
    for ticker in sector_tickers:
        # Sharpe Ratio para 2 años
        mean_return_2yr = returns_2yr[ticker].mean() * 252
        volatility_2yr = returns_2yr[ticker].std() * np.sqrt(252)
        sharpe_2yr = (mean_return_2yr - risk_free_rate) / volatility_2yr

        # Sharpe Ratio para 5 años
        mean_return_5yr = returns_5yr[ticker].mean() * 252
        volatility_5yr = returns_5yr[ticker].std() * np.sqrt(252)
        sharpe_5yr = (mean_return_5yr - risk_free_rate) / volatility_5yr

        sharpe_ratios[ticker] = (sharpe_2yr, sharpe_5yr)

    # Obtener lista de CEDEARs
    cedears = get_cedears()

    # 4. Crear el gráfico cuadrante con punto de referencia
    sharpe_df = pd.DataFrame(sharpe_ratios, index=["2 Years", "5 Years"]).T
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 7), dpi=120)

    # Scatter de los tickers del sector especificado
    for ticker in sharpe_df.index:
        if ticker in cedears:
            ax.scatter(sharpe_df.loc[ticker, "2 Years"], sharpe_df.loc[ticker, "5 Years"], s=200, color="magenta", alpha=0.7, label=f"{ticker} (CEDEAR)")  # Color para CEDEAR
        else:
            ax.scatter(sharpe_df.loc[ticker, "2 Years"], sharpe_df.loc[ticker, "5 Years"], s=200, color="cyan", alpha=0.3, label=f"{ticker} (Stock)")  # Color para acciones normales

    # Agregar el punto de referencia del S&P 500
    ax.scatter(sp500_sharpe_ratio_2yr, sp500_sharpe_ratio_5yr, color="red", alpha=0.5, s=550, label="S&P 500")
    ax.text(sp500_sharpe_ratio_2yr, sp500_sharpe_ratio_5yr, "SP500", color="white", fontsize=8, ha="center", va="center")

    # Añadir etiquetas a los puntos en el scatter
    for ticker in sharpe_df.index:
        ax.text(sharpe_df.loc[ticker, "2 Years"], sharpe_df.loc[ticker, "5 Years"], ticker, color="white", fontsize=6, alpha=0.99, ha="center", va="center")

    # Añadir líneas punteadas a través del punto de referencia del S&P 500
    ax.axhline(sp500_sharpe_ratio_5yr.item(), color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(sp500_sharpe_ratio_2yr.item(), color="gray", linestyle="--", linewidth=0.8)

    # Filtrar valores NaN o Inf en los datos de '2 Years' y '5 Years'
    sharpe_df = sharpe_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["2 Years", "5 Years"])

    # Recalcular x e y después del filtrado
    x = sharpe_df["2 Years"]
    y = sharpe_df["5 Years"]

    # Ordenar los valores de 'x' para evitar problemas con la línea
    sorted_indices = np.argsort(x)
    x_sorted = x.iloc[sorted_indices]
    y_sorted = y.iloc[sorted_indices]

    # Calcular la regresión lineal
    slope, intercept = np.polyfit(x_sorted, y_sorted, 1)
    regression_line = slope * x_sorted + intercept

    # Graficar la línea de regresión como línea punteada en rojo
    ax.plot(x_sorted, regression_line, color="red", linestyle="--", linewidth=1.5, label="Línea de Regresión", alpha=0.5)

    # Configuración del gráfico
    ax.set_xlabel("Sharpe Ratio (2 Years)", color="white")
    ax.set_ylabel("Sharpe Ratio (5 Years)", color="white")
    ax.set_title(f"{sector_name} Sector Risk Quadrant (Sharpe Ratio)", color="white")
    plt.show()

# Llamar a la función para analizar el sector financiero, puedes cambiar el nombre del sector aquí
analyze_sector_with_sharpe("Health Care")  # Cambia "Financials" por cualquier otro sector
