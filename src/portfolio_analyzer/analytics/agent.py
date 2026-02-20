from __future__ import annotations

import importlib


def run_portfolio_agent(
    ticker: str,
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
):
    """Execute the existing portfolio agent and return its decision payload."""
    portfolio_agent_module = importlib.import_module("ai_agent.pages.01_portfolio_agent")
    run_hedge_fund = portfolio_agent_module.run_hedge_fund
    return run_hedge_fund(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=show_reasoning,
    )
