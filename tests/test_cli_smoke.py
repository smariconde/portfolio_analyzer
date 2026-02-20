from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_analyzer.cli.main import build_parser


def test_cli_has_expected_top_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["ui"])
    assert args.command == "ui"


def test_cli_sortino_subcommand_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "sortino", "--no-pdf"])
    assert args.command == "run"
    assert args.module == "sortino"
    assert args.no_pdf is True


def test_cli_optimizer_subcommand_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "optimizer", "--tickers", "AAPL,MSFT"])
    assert args.command == "run"
    assert args.module == "optimizer"
    assert args.tickers == "AAPL,MSFT"
