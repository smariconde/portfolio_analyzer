import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms

def create_optimal_portfolio_with_genetic(tickers, portfolio_size, cash_percentage):
    # Descargar precios de cierre ajustados de los últimos 5 años
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Calcular el rendimiento esperado y la matriz de covarianza
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    # Configuración del algoritmo genético
    num_assets = len(tickers)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizar el Sortino Ratio
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 1)  # Pesos iniciales entre 0 y 1
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        weights = np.array(individual)
        weights /= weights.sum()  # Normalizar los pesos para que sumen 1
        
        # Calcular rendimiento y riesgo
        portfolio_return = np.dot(mu, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        
        # Calcular Sortino Ratio
        portfolio_returns = np.dot(data.pct_change().dropna(), weights)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_risk = np.sqrt(np.mean(downside_returns**2))
        sortino_ratio = portfolio_return / downside_risk if downside_risk > 0 else 0
        return sortino_ratio,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Configuración del algoritmo
    population = toolbox.population(n=100)  # Tamaño de la población
    num_generations = 200  # Número de generaciones

    # Ejecutar el algoritmo genético
    result, _ = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations, verbose=False)

    # Obtener los mejores pesos
    best_individual = tools.selBest(population, k=1)[0]
    optimal_weights = np.array(best_individual) / np.sum(best_individual)
    
    # Calcular inversión
    total_investment = portfolio_size * (1 - cash_percentage)
    investment_per_asset = optimal_weights * total_investment

    # Crear un DataFrame con los resultados
    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": optimal_weights,
        "Investment": investment_per_asset
    })
    weights_df = weights_df[weights_df['Weight'] > 0]  # Filtrar activos con peso > 0

    # Mostrar resultados
    print("Pesos óptimos del portafolio:")
    print(weights_df)
    return weights_df

# Configuración del portafolio
tickers = ["BRK-B", "GOOGL", "MSFT", "LRCX", "LLY", "MELI", "VIST", "NU", "COST"]
portfolio_size = 40000  # Tamaño total del portafolio en USD
cash_percentage = 0.6   # Porcentaje de efectivo no invertido

# Generar portafolio óptimo
weights_df = create_optimal_portfolio_with_genetic(tickers, portfolio_size, cash_percentage)

