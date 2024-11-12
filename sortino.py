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

def analyze_sector_with_sortino(sector_name):
    # 1. Scraping de Wikipedia para obtener los tickers del S&P 500 y sus sectores
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]  # La primera tabla en la página suele contener la lista de empresas

    # Filtrar empresas según el sector proporcionado
    tickers = sp500_table[sp500_table["GICS Sector"] == sector_name]["Symbol"].tolist()

    sector_tickers = ['BRK-B' if ticker == 'BRK.B' else ticker for ticker in tickers]

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

    # Cálculo del Sortino Ratio para el S&P 500 a 2 años y a 5 años
    downside_risk_2yr_sp500 = np.sqrt((sp500_returns[sp500_returns.index >= two_years_ago_str] ** 2).mean())
    sortino_2yr_sp500 = (sp500_returns[sp500_returns.index >= two_years_ago_str].mean() * 252 - risk_free_rate) / downside_risk_2yr_sp500

    downside_risk_5yr_sp500 = np.sqrt((sp500_returns ** 2).mean())
    sortino_5yr_sp500 = (sp500_returns.mean() * 252 - risk_free_rate) / downside_risk_5yr_sp500

    # 2. Calcular retornos diarios
    returns = data.pct_change()

    # 3. Dividir los datos en ventanas de 2 años y 5 años
    returns_2yr = returns[returns.index >= two_years_ago_str]
    returns_5yr = returns

    # Cálculo del Sortino Ratio para 2 y 5 años
    sortino_ratios = {}
    for ticker in sector_tickers:
        # Sortino Ratio para 2 años
        downside_risk_2yr_ticker = np.sqrt((returns_2yr[ticker][returns_2yr[ticker] < 0] ** 2).mean())
        sortino_2yr_ticker = (returns_2yr[ticker].mean() * 252 - risk_free_rate) / downside_risk_2yr_ticker

        # Sortino Ratio para 5 años
        downside_risk_5yr_ticker = np.sqrt((returns_5yr[ticker][returns_5yr[ticker] < 0] ** 2).mean())
        sortino_5yr_ticker = (returns_5yr[ticker].mean() * 252 - risk_free_rate) / downside_risk_5yr_ticker

        sortino_ratios[ticker] = (sortino_2yr_ticker, sortino_5yr_ticker)

    # Obtener lista de CEDEARs
    cedears = get_cedears()

    # 4. Crear el gráfico cuadrante con punto de referencia
    sortino_df = pd.DataFrame(sortino_ratios, index=["2 Years", "5 Years"]).T
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 7), dpi=120)

    # Scatter de los tickers del sector especificado
    for ticker in sortino_df.index:
        if ticker in cedears:
            ax.scatter(sortino_df.loc[ticker, "2 Years"], sortino_df.loc[ticker, "5 Years"], s=200, color="magenta", alpha=0.7, label=f"{ticker} (CEDEAR)")  # Color para CEDEAR
        else:
            ax.scatter(sortino_df.loc[ticker, "2 Years"], sortino_df.loc[ticker, "5 Years"], s=200, color="cyan", alpha=0.3, label=f"{ticker} (Stock)")  # Color para acciones normales

    # Agregar el punto de referencia del S&P 500
    ax.scatter(sortino_2yr_sp500, sortino_5yr_sp500, color="red", alpha=0.5, s=550, label="S&P 500")
    ax.text(sortino_2yr_sp500, sortino_5yr_sp500, "SP500", color="white", fontsize=8, ha="center", va="center")

    # Añadir etiquetas a los puntos en el scatter
    for ticker in sortino_df.index:
        ax.text(sortino_df.loc[ticker, "2 Years"], sortino_df.loc[ticker, "5 Years"], ticker, color="white", fontsize=6, alpha=0.99, ha="center", va="center")

    # Añadir líneas punteadas a través del punto de referencia del S&P 500
    ax.axhline(sortino_5yr_sp500, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(sortino_2yr_sp500, color="gray", linestyle="--", linewidth=0.8)

    # Filtrar valores NaN o Inf en los datos de '2 Years' y '5 Years'
    sortino_df = sortino_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["2 Years", "5 Years"])

    # Recalcular x e y después del filtrado
    x = sortino_df["2 Years"]
    y = sortino_df["5 Years"]

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
    ax.set_xlabel("Sortino Ratio (2 Years)", color="white")
    ax.set_ylabel("Sortino Ratio (5 Years)", color="white")
    ax.set_title(f"{sector_name} Sector Risk Quadrant (Sortino Ratio)", color="white")
    plt.show()

# Llamar a la función para analizar el sector financiero, puedes cambiar el nombre del sector aquí
analyze_sector_with_sortino("Consumer Staples")  # Cambia "Financials" por cualquier otro sector

