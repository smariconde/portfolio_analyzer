#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


METHOD_BY_ASSET_TYPE = {
    "financial": "dividend_discount",
    "stable-cashflow": "fcff_dcf",
    "high-growth": "fcff_dcf_two_stage",
    "cyclical": "fcff_dcf_normalized_cycle",
    "commodity": "fcff_dcf_normalized_cycle",
    "no-earnings": "relative_valuation_with_fcff_bridge",
}


REQUIRED_ASSUMPTIONS = {
    "dividend_discount": [
        "cost_of_equity",
        "dividend_payout_ratio",
        "growth_rate",
        "terminal_growth_rate",
    ],
    "fcff_dcf": [
        "revenue_growth_path",
        "operating_margin_path",
        "tax_rate",
        "reinvestment_rate",
        "wacc",
        "terminal_growth_rate",
        "net_debt",
        "shares_outstanding",
    ],
    "fcff_dcf_two_stage": [
        "high_growth_years",
        "revenue_growth_path",
        "operating_margin_path",
        "tax_rate",
        "reinvestment_rate",
        "wacc",
        "terminal_growth_rate",
        "net_debt",
        "shares_outstanding",
    ],
    "fcff_dcf_normalized_cycle": [
        "normalized_revenue",
        "normalized_operating_margin",
        "tax_rate",
        "cycle_assumption_years",
        "reinvestment_rate",
        "wacc",
        "terminal_growth_rate",
        "net_debt",
        "shares_outstanding",
    ],
    "relative_valuation_with_fcff_bridge": [
        "peer_multiple_name",
        "peer_multiple_value",
        "fundamental_metric",
        "uncertainty_discount_pct",
        "bridge_to_positive_fcff_year",
    ],
}


def select_method(asset_type: str, has_dividends: bool) -> str:
    normalized = asset_type.strip().lower()
    method = METHOD_BY_ASSET_TYPE.get(normalized, "fcff_dcf")
    if normalized == "financial" and not has_dividends:
        return "fcff_dcf"
    return method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select intrinsic valuation method and required assumptions by asset type."
    )
    parser.add_argument("--ticker", required=True, help="Ticker or symbol")
    parser.add_argument(
        "--asset-type",
        required=True,
        choices=[
            "financial",
            "stable-cashflow",
            "high-growth",
            "cyclical",
            "commodity",
            "no-earnings",
        ],
        help="Asset profile for valuation method selection",
    )
    parser.add_argument("--sector", default="", help="Sector label")
    parser.add_argument("--country", default="", help="Country label")
    parser.add_argument(
        "--has-dividends",
        action="store_true",
        help="Set for financial firms with stable dividend policy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method = select_method(args.asset_type, args.has_dividends)
    output = {
        "ticker": args.ticker.upper().strip(),
        "asset_type": args.asset_type,
        "sector": args.sector,
        "country": args.country,
        "recommended_method": method,
        "required_assumptions": REQUIRED_ASSUMPTIONS.get(method, REQUIRED_ASSUMPTIONS["fcff_dcf"]),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
