from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_analyzer.analytics.optimizer import optimize_portfolio


def main() -> int:
    parser = argparse.ArgumentParser(description="Run portfolio optimization")
    parser.add_argument("--tickers", default="BRK-B,GOOGL,MSFT,LRCX,LLY,MELI,VIST,NU,COST")
    parser.add_argument("--portfolio-size", type=float, default=40000)
    parser.add_argument("--cash-percentage", type=float, default=0.6)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    result = optimize_portfolio(
        tickers=tickers,
        portfolio_size=args.portfolio_size,
        cash_percentage=args.cash_percentage,
    )

    print("Distribucion optima del portafolio:")
    print(result.weights.to_string())
    print(f"Sortino Portfolio: {result.sortino_ratio:.2f}")
    print(f"Sortino S&P 500: {result.benchmark_sortino:.2f}")
    print(f"Sharpe S&P 500: {result.benchmark_sharpe:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
