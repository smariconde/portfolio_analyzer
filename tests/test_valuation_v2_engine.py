import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_agent.valuation_v2.engine import run_valuation
from ai_agent.valuation_v2.schemas import validate_output_schema

os.environ["VALUATION_ONLINE_ASSUMPTIONS"] = "0"


def _base_metrics() -> dict:
    return {
        "totalRevenue": 100_000_000_000,
        "sharesOutstanding": 15_000_000_000,
        "currentPrice": 180.0,
        "operating_margin": 0.30,
        "profitMargins": 0.24,
        "revenue_growth": 0.08,
        "price_to_sales_ratio": 6.0,
        "earnings_per_share": 6.5,
        "dividendRate": 1.0,
        "totalDebt": 110_000_000_000,
        "totalCash": 60_000_000_000,
        "beta": 1.1,
    }


def test_v2_output_schema_is_valid():
    payload = run_valuation(
        ticker="AAPL",
        metrics=_base_metrics(),
        asset_type="stable-cashflow",
        country="US",
        strict_mode=True,
    )
    ok, errors = validate_output_schema(payload)
    assert ok, errors


def test_wacc_constraint_is_enforced():
    payload = run_valuation(
        ticker="TEST",
        metrics=_base_metrics(),
        asset_type="stable-cashflow",
        country="US",
        llm_assumptions={"wacc": 0.03, "terminal_growth_rate": 0.04},
        strict_mode=True,
    )
    checks = payload["diagnostics"]["checks"]
    assert any(check["name"] == "wacc_gt_terminal_growth" for check in checks)
    assert payload["assumptions"]["financing"]["wacc"] > payload["assumptions"]["terminal"]["terminal_growth_rate"]


def test_shares_must_be_positive():
    metrics = _base_metrics()
    metrics["sharesOutstanding"] = 0
    try:
        run_valuation(
            ticker="TEST",
            metrics=metrics,
            asset_type="stable-cashflow",
            country="US",
        )
    except ValueError as exc:
        assert "shares_outstanding must be > 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError when sharesOutstanding is zero")


def test_intrinsic_decreases_when_wacc_increases():
    low_wacc = run_valuation(
        ticker="TEST",
        metrics=_base_metrics(),
        asset_type="stable-cashflow",
        country="US",
        llm_assumptions={"wacc": 0.08, "terminal_growth_rate": 0.03},
        strict_mode=True,
    )
    high_wacc = run_valuation(
        ticker="TEST",
        metrics=_base_metrics(),
        asset_type="stable-cashflow",
        country="US",
        llm_assumptions={"wacc": 0.12, "terminal_growth_rate": 0.03},
        strict_mode=True,
    )
    assert high_wacc["result"]["intrinsic_value_per_share"] < low_wacc["result"]["intrinsic_value_per_share"]


def test_method_override_and_selection_reason_are_exposed():
    payload = run_valuation(
        ticker="TEST",
        metrics=_base_metrics(),
        asset_type="stable-cashflow",
        country="US",
        method_override="fcff_dcf_two_stage",
        method_selection_reason="LLM selected high-growth profile with margin convergence.",
        method_selected_by="llm",
    )
    assert payload["valuation_method"] == "fcff_dcf_two_stage"
    assert payload["method_selection"]["selected_by"] == "llm"
    assert "LLM selected high-growth profile" in payload["method_selection"]["reason"]


def test_normalized_cycle_differs_from_plain_fcff():
    base = _base_metrics()
    cyclical = run_valuation(
        ticker="TEST",
        metrics=base,
        asset_type="cyclical",
        country="US",
        method_override="fcff_dcf_normalized_cycle",
        llm_assumptions={"wacc": 0.1, "terminal_growth_rate": 0.03},
        monte_carlo_trials=0,
    )
    plain = run_valuation(
        ticker="TEST",
        metrics=base,
        asset_type="stable-cashflow",
        country="US",
        method_override="fcff_dcf",
        llm_assumptions={"wacc": 0.1, "terminal_growth_rate": 0.03},
        monte_carlo_trials=0,
    )
    assert cyclical["result"]["intrinsic_value_per_share"] != plain["result"]["intrinsic_value_per_share"]


def test_strict_mode_policy_raise_fails_on_critical_constraint():
    failed = False
    try:
        run_valuation(
            ticker="TEST",
            metrics=_base_metrics(),
            asset_type="stable-cashflow",
            country="US",
            llm_assumptions={"wacc": 0.05, "terminal_growth_rate": 0.049},
            strict_mode_policy="raise",
        )
    except ValueError as exc:
        failed = "wacc_gt_terminal_growth" in str(exc)
    assert failed


def test_monte_carlo_range_is_present_and_ordered():
    payload = run_valuation(
        ticker="TEST",
        metrics=_base_metrics(),
        asset_type="stable-cashflow",
        country="US",
        monte_carlo_trials=300,
        monte_carlo_seed=7,
    )
    vr = payload["result"]["valuation_range"]
    assert vr["p10"] <= vr["p50"] <= vr["p90"]


def test_financial_model_returns_cross_checks():
    payload = run_valuation(
        ticker="JPM",
        metrics=_base_metrics(),
        asset_type="financial",
        country="US",
        method_override="dividend_discount",
        monte_carlo_trials=0,
    )
    assert payload["valuation_method"] == "dividend_discount"
    assert "ddm_intrinsic_value_per_share" in payload["cross_checks"]
    assert "residual_income_intrinsic_value_per_share" in payload["cross_checks"]
