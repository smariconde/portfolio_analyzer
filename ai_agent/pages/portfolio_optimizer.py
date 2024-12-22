import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms


def create_deap_classes():
    """Creates the DEAP Fitness and Individual classes."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)



def create_optimal_portfolio_with_genetic(tickers, portfolio_size, cash_percentage):
    """
    Calculates optimal portfolio weights using a genetic algorithm.

    Args:
        tickers (list): List of stock tickers.
        portfolio_size (float): Total portfolio size in USD.
        cash_percentage (float): Percentage of portfolio held in cash (0-1).

    Returns:
        pandas.DataFrame: DataFrame with optimal weights, investment per asset.
    """
    if not tickers:
        st.error("Please enter at least one ticker.")
        return pd.DataFrame()

    # Download adjusted close prices for the last 5 years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    except Exception as e:
        st.error(f"Error downloading data for tickers: {e}")
        return pd.DataFrame()

    if data.empty:
        st.error("No data found for the given tickers. Please check the ticker symbols.")
        return pd.DataFrame()

    # Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    
    # Genetic algorithm setup
    num_assets = len(tickers)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate(individual):
        weights = np.array(individual)
        weights /= weights.sum()

        # Calculate portfolio return and risk
        portfolio_return = np.dot(mu, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(S, weights)))

        # Calculate Sortino Ratio
        portfolio_returns = np.dot(data.pct_change().dropna(), weights)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_risk = np.sqrt(np.mean(downside_returns**2))
        sortino_ratio = portfolio_return / downside_risk if downside_risk > 0 else 0
        return sortino_ratio,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    

    # Algorithm parameters
    population = toolbox.population(n=100)
    num_generations = 200

    # Run the genetic algorithm
    result, _ = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_generations, verbose=False)

    # Get the best weights
    best_individual = tools.selBest(population, k=1)[0]
    optimal_weights = np.array(best_individual) / np.sum(best_individual)

    # Calculate investment
    total_investment = portfolio_size * (1 - cash_percentage)
    investment_per_asset = optimal_weights * total_investment

    # Create a DataFrame with results
    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": optimal_weights,
        "Investment": investment_per_asset
    })
    weights_df = weights_df[weights_df['Weight'] > 0]

    return weights_df


def main():
    st.title("Optimal Portfolio Optimizer")

    portfolio_size = st.number_input("Portfolio Size (USD)", min_value=1000, value=40000, step=1000)
    cash_percentage = st.slider("Cash Percentage", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

    # Tickers input
    if "tickers" not in st.session_state:
        st.session_state.tickers = []

    new_ticker = st.text_input("Enter a stock ticker to add to the portfolio", "")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Add Ticker"):
            if new_ticker:
                st.session_state.tickers.append(new_ticker.strip().upper())
                new_ticker = ""
    with col2:
        if st.button("Remove Last Ticker"):
            if st.session_state.tickers:
                st.session_state.tickers.pop()

    if st.session_state.tickers:
        st.write("Current Tickers:")
        st.write(st.session_state.tickers)

    if st.button("Calculate Portfolio"):
        with st.spinner("Calculating optimal portfolio..."):
            weights_df = create_optimal_portfolio_with_genetic(st.session_state.tickers, portfolio_size, cash_percentage)

            if not weights_df.empty:
                st.subheader("Optimal Portfolio Allocation")
                st.dataframe(weights_df)

                total_investment = portfolio_size * (1 - cash_percentage)
                total_invested = weights_df['Investment'].sum()
                total_cash = portfolio_size - total_invested

                st.metric("Total Invested", f"${total_invested:,.2f}")
                st.metric("Total Cash", f"${total_cash:,.2f}")
            else:
                st.warning("No optimal portfolio could be calculated. Please check your inputs.")

if __name__ == "__main__":
     # Create the DEAP classes only once
    create_deap_classes()
    main()