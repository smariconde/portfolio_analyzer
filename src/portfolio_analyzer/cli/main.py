from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="portfolio-analyzer",
        description="Unified finance toolkit CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ui = subparsers.add_parser("ui", help="Run Streamlit app")
    ui.add_argument("--app", default="ai_agent/Home.py", help="Path to streamlit entrypoint")

    run = subparsers.add_parser("run", help="Run analytics workflows")
    run_sub = run.add_subparsers(dest="module", required=True)

    sortino = run_sub.add_parser("sortino", help="Run Sortino analysis")
    sortino.add_argument("--no-pdf", action="store_true", help="Skip PDF report generation")

    optimizer = run_sub.add_parser("optimizer", help="Run portfolio optimization")
    optimizer.add_argument(
        "--tickers",
        required=True,
        help="Comma separated tickers, e.g. AAPL,MSFT",
    )
    optimizer.add_argument("--portfolio-size", type=float, default=40000)
    optimizer.add_argument("--cash-percentage", type=float, default=0.6)

    commodities = run_sub.add_parser("commodities", help="Run soybean spread analysis")
    commodities.add_argument("--start-date", default="2024-01-01")
    commodities.add_argument("--prices-chart", default="soja_precios_diario_2024.png")
    commodities.add_argument("--spread-chart", default="soja_brecha_final.png")

    agent = run_sub.add_parser("agent", help="Run portfolio agent once")
    agent.add_argument("--ticker", required=True)
    agent.add_argument("--start-date", required=True)
    agent.add_argument("--end-date", required=True)
    agent.add_argument("--cash", type=float, default=5000)
    agent.add_argument("--invested", type=float, default=20000)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ui":
        subprocess.run(["streamlit", "run", args.app], check=False)
        return 0

    if args.command == "run" and args.module == "sortino":
        from portfolio_analyzer.analytics.sortino import run_sortino_analysis

        result = run_sortino_analysis(generate_pdf=not args.no_pdf)
        print(
            json.dumps(
                {
                    "output_csv": str(result.output_csv),
                    "output_txt": str(result.output_txt),
                    "output_pdf": str(result.output_pdf) if result.output_pdf else None,
                    "total_tickers": result.total_tickers,
                    "total_sectors": result.total_sectors,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run" and args.module == "optimizer":
        from portfolio_analyzer.analytics.optimizer import optimize_portfolio

        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        result = optimize_portfolio(
            tickers=tickers,
            portfolio_size=args.portfolio_size,
            cash_percentage=args.cash_percentage,
        )
        print(result.weights.to_string())
        print(f"sortino_ratio={result.sortino_ratio:.4f}")
        print(f"benchmark_sortino={result.benchmark_sortino:.4f}")
        print(f"benchmark_sharpe={result.benchmark_sharpe:.4f}")
        return 0

    if args.command == "run" and args.module == "commodities":
        from portfolio_analyzer.data.commodities import fetch_soybean_spread, save_default_charts

        result = fetch_soybean_spread(start_date=args.start_date)
        save_default_charts(result.dataframe, args.prices_chart, args.spread_chart)
        print(
            json.dumps(
                {
                    "correlation": result.correlation,
                    "spread_pct_latest": result.spread_pct_latest,
                    "spread_pct_mean": result.spread_pct_mean,
                    "prices_chart": str(Path(args.prices_chart)),
                    "spread_chart": str(Path(args.spread_chart)),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "run" and args.module == "agent":
        from portfolio_analyzer.analytics.agent import run_portfolio_agent

        portfolio = {"cash": args.cash, "stock": 0, "portfolio_value": args.invested}
        output = run_portfolio_agent(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            portfolio=portfolio,
            show_reasoning=False,
        )
        print(output)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
