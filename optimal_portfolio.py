import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from datetime import datetime, timedelta

def create_optimal_portfolio_with_sortino(tickers, portfolio_size, cash_percentage, min_weight, max_weight):
    # Descargar precios de cierre ajustados de los últimos 5 años
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Descargar datos del S&P 500 como benchmark
    sp500_data = yf.download("^GSPC", start=start_date, end=end_date)['Adj Close']
    
    # Calcular el rendimiento esperado y la matriz de covarianza para el portafolio
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    
    # Crear el objeto de frontera eficiente
    ef = EfficientFrontier(mu, S)

    # Establecer límites para cada peso (por ejemplo, un mínimo de 5% y un máximo de 40%)
    min_weight = 0.05
    max_weight = 0.40
    ef.add_constraint(lambda w: w >= min_weight)
    ef.add_constraint(lambda w: w <= max_weight)

    # Optimizar para el Sortino Ratio
    risk_aversion = 2  # Ajustar según la preferencia de riesgo
    weights = ef.max_quadratic_utility(risk_aversion=risk_aversion, market_neutral=False)
    cleaned_weights = ef.clean_weights()

    # Calcular la inversión excluyendo el porcentaje de efectivo
    total_investment = portfolio_size * (1 - cash_percentage)
    weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
    weights_df['Investment'] = weights_df['Weight'] * total_investment
    weights_df = weights_df[weights_df['Investment'] > 0]  # Filtrar activos con peso > 0
    
    # Mostrar distribución del portafolio
    print("Distribución óptima del portafolio:")
    print(weights_df)
    print(f"Total efectivo excluido: {portfolio_size * cash_percentage:.2f}")
    
    # Calcular el rendimiento del portafolio
    portfolio_returns = (data.pct_change().dropna() * weights_df['Weight'].values).sum(axis=1)
    
    # Cálculo del Sortino Ratio del portafolio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    sortino_ratio = portfolio_returns.mean() / downside_returns.std() * np.sqrt(252)
        
    # Desempeño del portafolio (Sharpe Ratio y otros métricos)
    performance = ef.portfolio_performance(verbose=True)

    print(f"Sortino Ratio: {sortino_ratio:.2f}")

    # Calcular métricas para el S&P 500
    sp500_returns = sp500_data.pct_change().dropna()
    sp500_annualized_return = sp500_returns.mean() * 252
    sp500_annualized_volatility = sp500_returns.std() * np.sqrt(252)
    sp500_sharpe_ratio = sp500_annualized_return / sp500_annualized_volatility

    # Calcular Sortino Ratio del S&P 500
    sp500_downside_returns = sp500_returns[sp500_returns < 0]
    sp500_downside_deviation = sp500_downside_returns.std() * np.sqrt(252)  # Ajuste anualizado
    sp500_sortino_ratio = sp500_annualized_return / sp500_downside_deviation if sp500_downside_deviation != 0 else np.nan

    # Mostrar métricas del S&P 500
    print("\nBenchmark (S&P 500) Performance:")
    print(f"Rendimiento anualizado S&P 500: {sp500_annualized_return:.2%}")
    print(f"Volatilidad anualizada S&P 500: {sp500_annualized_volatility:.2%}")
    print(f"Sharpe Ratio S&P 500: {sp500_sharpe_ratio:.2f}")
    print(f"Sortino Ratio S&P 500: {sp500_sortino_ratio:.2f}")
    
    return weights_df, performance, sortino_ratio, sp500_sharpe_ratio, sp500_sortino_ratio

# Configuración del portafolio
tickers = ["BRK-B", "CAT", "CNA", "MSFT", "AMAT", "LRCX", "LLY","MELI", "VIST", "NVO"]  # Lista de tickers de ejemplo
portfolio_size = 10000  # Tamaño total del portafolio en USD
cash_percentage = 0.6   # Porcentaje de efectivo no invertido
min_weight = 0.05
max_weight = 0.40

# Generar portafolio óptimo
weights_df, performance, sortino_ratio, sp500_sharpe_ratio, sp500_sortino_ratio = create_optimal_portfolio_with_sortino(
    tickers, portfolio_size, cash_percentage, min_weight, max_weight)




