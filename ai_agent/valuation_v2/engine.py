from __future__ import annotations

import math
import random
from typing import Any

from ai_agent.valuation_v2.market_assumptions import (
    LOCAL_LONG_RUN_NOMINAL_GROWTH,
    get_market_assumptions,
    market_assumptions_to_dict,
)


ASSET_TYPES = {
    "financial",
    "stable-cashflow",
    "high-growth",
    "cyclical",
    "commodity",
    "no-earnings",
}

METHOD_BY_ASSET_TYPE = {
    "financial": "dividend_discount",
    "stable-cashflow": "fcff_dcf",
    "high-growth": "fcff_dcf_two_stage",
    "cyclical": "fcff_dcf_normalized_cycle",
    "commodity": "fcff_dcf_normalized_cycle",
    "no-earnings": "relative_valuation_with_fcff_bridge",
}

ALLOWED_METHODS = {
    "dividend_discount",
    "residual_income",
    "fcff_dcf",
    "fcff_dcf_two_stage",
    "fcff_dcf_normalized_cycle",
    "relative_valuation_with_fcff_bridge",
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def select_method(asset_type: str, has_dividends: bool = False) -> str:
    normalized = (asset_type or "").strip().lower()
    if normalized not in ASSET_TYPES:
        normalized = "stable-cashflow"
    method = METHOD_BY_ASSET_TYPE[normalized]
    if normalized == "financial" and not has_dividends:
        return "fcff_dcf"
    return method


def _merge_assumptions(defaults: dict[str, float], proposed: dict[str, Any]) -> dict[str, float]:
    merged = dict(defaults)
    for key, value in (proposed or {}).items():
        if key in merged:
            merged[key] = _num(value, merged[key])
    return merged


def _validate_or_correct(
    condition: bool,
    *,
    name: str,
    details: str,
    strict_mode_policy: str,
    checks: list[dict[str, Any]],
) -> None:
    if condition:
        checks.append({"name": name, "passed": True, "severity": "critical", "details": "ok"})
        return
    checks.append({"name": name, "passed": False, "severity": "critical", "details": details})
    if strict_mode_policy == "raise":
        raise ValueError(f"{name}: {details}")


def _sanitize_assumptions(
    assumptions: dict[str, float],
    country: str,
    *,
    strict_mode_policy: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    sanitized = dict(assumptions)

    sanitized["revenue_growth"] = _clamp(sanitized["revenue_growth"], -0.2, 0.5)
    sanitized["earnings_growth"] = _clamp(sanitized["earnings_growth"], -0.25, 0.5)
    sanitized["operating_margin"] = _clamp(sanitized["operating_margin"], -0.1, 0.7)
    sanitized["normalized_operating_margin"] = _clamp(sanitized["normalized_operating_margin"], -0.05, 0.5)
    sanitized["tax_rate"] = _clamp(sanitized["tax_rate"], 0.0, 0.5)
    sanitized["reinvestment_rate"] = _clamp(sanitized["reinvestment_rate"], 0.0, 0.95)
    sanitized["wacc"] = _clamp(sanitized["wacc"], 0.04, 0.35)
    sanitized["cost_of_equity"] = _clamp(sanitized["cost_of_equity"], 0.05, 0.45)
    sanitized["cost_of_debt"] = _clamp(sanitized["cost_of_debt"], 0.0, 0.25)
    sanitized["target_debt_weight"] = _clamp(sanitized["target_debt_weight"], 0.0, 0.9)
    sanitized["terminal_growth_rate"] = _clamp(sanitized["terminal_growth_rate"], 0.0, 0.08)
    sanitized["payout_ratio"] = _clamp(sanitized["payout_ratio"], 0.0, 0.95)
    sanitized["peer_multiple_value"] = _clamp(sanitized["peer_multiple_value"], 0.1, 40.0)
    sanitized["uncertainty_discount_pct"] = _clamp(sanitized["uncertainty_discount_pct"], 0.0, 0.75)
    sanitized["shares_reduction_per_year"] = _clamp(sanitized["shares_reduction_per_year"], -0.08, 0.12)
    sanitized["high_growth_years"] = _clamp(sanitized["high_growth_years"], 3.0, 10.0)
    sanitized["cycle_normalization_factor"] = _clamp(sanitized["cycle_normalization_factor"], 0.7, 1.3)
    sanitized["bridge_to_positive_fcff_year"] = _clamp(sanitized["bridge_to_positive_fcff_year"], 1.0, 8.0)
    sanitized["steady_state_roic"] = _clamp(sanitized["steady_state_roic"], 0.03, 0.4)

    long_run_cap = LOCAL_LONG_RUN_NOMINAL_GROWTH.get((country or "US").strip().upper(), 0.03) + 0.005
    if sanitized["terminal_growth_rate"] > long_run_cap:
        msg = (
            f"terminal_growth_rate reduced from {sanitized['terminal_growth_rate']:.4f} "
            f"to long-run cap {long_run_cap:.4f}"
        )
        checks.append({"name": "terminal_growth_cap", "passed": False, "severity": "warning", "details": msg})
        if strict_mode_policy != "raise":
            sanitized["terminal_growth_rate"] = long_run_cap
    else:
        checks.append({"name": "terminal_growth_cap", "passed": True, "severity": "warning", "details": "ok"})

    wacc_gap = sanitized["wacc"] - sanitized["terminal_growth_rate"]
    if wacc_gap <= 0.01:
        msg = (
            f"wacc ({sanitized['wacc']:.4f}) must exceed terminal_growth_rate "
            f"({sanitized['terminal_growth_rate']:.4f}) by at least 100 bps"
        )
        checks.append({"name": "wacc_gt_terminal_growth", "passed": False, "severity": "critical", "details": msg})
        if strict_mode_policy == "raise":
            raise ValueError(f"wacc_gt_terminal_growth: {msg}")
        sanitized["wacc"] = sanitized["terminal_growth_rate"] + 0.02
    else:
        checks.append({"name": "wacc_gt_terminal_growth", "passed": True, "severity": "critical", "details": "ok"})

    ke_gap = sanitized["cost_of_equity"] - sanitized["terminal_growth_rate"]
    if ke_gap <= 0.01:
        msg = (
            f"cost_of_equity ({sanitized['cost_of_equity']:.4f}) must exceed terminal_growth_rate "
            f"({sanitized['terminal_growth_rate']:.4f}) by at least 100 bps"
        )
        checks.append({"name": "ke_gt_terminal_growth", "passed": False, "severity": "critical", "details": msg})
        if strict_mode_policy == "raise":
            raise ValueError(f"ke_gt_terminal_growth: {msg}")
        sanitized["cost_of_equity"] = sanitized["terminal_growth_rate"] + 0.02
    else:
        checks.append({"name": "ke_gt_terminal_growth", "passed": True, "severity": "critical", "details": "ok"})

    min_reinvestment = max(0.0, sanitized["terminal_growth_rate"] / max(sanitized["steady_state_roic"], 1e-6))
    if sanitized["reinvestment_rate"] + 0.02 < min_reinvestment:
        msg = (
            f"reinvestment_rate {sanitized['reinvestment_rate']:.4f} inconsistent with growth/ROIC; "
            f"minimum implied {min_reinvestment:.4f}"
        )
        checks.append({"name": "reinvestment_consistency", "passed": False, "severity": "critical", "details": msg})
        if strict_mode_policy == "raise":
            raise ValueError(f"reinvestment_consistency: {msg}")
        sanitized["reinvestment_rate"] = min(0.95, min_reinvestment)
    else:
        checks.append({"name": "reinvestment_consistency", "passed": True, "severity": "critical", "details": "ok"})

    return sanitized, checks


def _scenario_assumptions(base: dict[str, float], name: str) -> dict[str, float]:
    scenario = dict(base)
    if name == "bear":
        scenario["revenue_growth"] *= 0.75
        scenario["earnings_growth"] *= 0.75
        scenario["operating_margin"] -= 0.03
        scenario["normalized_operating_margin"] -= 0.02
        scenario["wacc"] += 0.015
        scenario["cost_of_equity"] += 0.015
        scenario["reinvestment_rate"] += 0.05
        scenario["terminal_growth_rate"] -= 0.005
        scenario["peer_multiple_value"] *= 0.8
    elif name == "bull":
        scenario["revenue_growth"] *= 1.2
        scenario["earnings_growth"] *= 1.15
        scenario["operating_margin"] += 0.02
        scenario["normalized_operating_margin"] += 0.015
        scenario["wacc"] -= 0.01
        scenario["cost_of_equity"] -= 0.01
        scenario["reinvestment_rate"] -= 0.03
        scenario["terminal_growth_rate"] += 0.003
        scenario["peer_multiple_value"] *= 1.1
    return scenario


def _compute_fcff_path(
    initial_revenue: float,
    growth_path: list[float],
    margin_path: list[float],
    tax_rate: float,
    reinvestment_rate: float,
) -> list[float]:
    revenue = initial_revenue
    fcff_values: list[float] = []
    for i, growth in enumerate(growth_path):
        revenue = revenue * (1.0 + growth)
        margin = margin_path[i] if i < len(margin_path) else margin_path[-1]
        ebit = revenue * margin
        nopat = ebit * (1.0 - tax_rate)
        fcff_values.append(nopat * (1.0 - reinvestment_rate))
    return fcff_values


def _discount_cashflows(cashflows: list[float], rate: float) -> float:
    total = 0.0
    for i, value in enumerate(cashflows, start=1):
        total += value / ((1.0 + rate) ** i)
    return total


def _run_fcff(
    *,
    initial_revenue: float,
    revenue_growth_path: list[float],
    operating_margin_path: list[float],
    tax_rate: float,
    reinvestment_rate: float,
    wacc: float,
    terminal_growth_rate: float,
    net_debt: float,
    shares_outstanding: float,
) -> dict[str, float]:
    fcff_values = _compute_fcff_path(
        initial_revenue,
        revenue_growth_path,
        operating_margin_path,
        tax_rate,
        reinvestment_rate,
    )
    terminal_fcf = fcff_values[-1] * (1.0 + terminal_growth_rate)
    terminal_value = terminal_fcf / (wacc - terminal_growth_rate)
    enterprise_value = _discount_cashflows(fcff_values, wacc) + (
        terminal_value / ((1.0 + wacc) ** len(fcff_values))
    )
    equity_value = enterprise_value - net_debt
    intrinsic_per_share = equity_value / shares_outstanding
    return {
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value_per_share": intrinsic_per_share,
    }


def _run_dividend_discount(
    *,
    eps: float,
    dividend_rate: float,
    payout_ratio: float,
    earnings_growth: float,
    cost_of_equity: float,
    terminal_growth_rate: float,
    horizon: int = 5,
) -> dict[str, float]:
    implied_dividend = max(0.0, eps) * payout_ratio
    d0 = max(dividend_rate, implied_dividend)
    dividends: list[float] = []
    current = d0
    growth = _clamp(earnings_growth * 0.7, -0.2, 0.2)
    for _ in range(horizon):
        current = current * (1.0 + growth)
        dividends.append(current)
    pv_dividends = _discount_cashflows(dividends, cost_of_equity)
    terminal_div = dividends[-1] * (1.0 + terminal_growth_rate)
    terminal_value = terminal_div / (cost_of_equity - terminal_growth_rate)
    intrinsic_per_share = pv_dividends + (terminal_value / ((1.0 + cost_of_equity) ** horizon))
    return {
        "enterprise_value": intrinsic_per_share,
        "equity_value": intrinsic_per_share,
        "intrinsic_value_per_share": intrinsic_per_share,
    }


def _run_residual_income(
    *,
    current_price: float,
    price_to_book: float,
    eps: float,
    payout_ratio: float,
    cost_of_equity: float,
    terminal_growth_rate: float,
    horizon: int = 5,
) -> dict[str, float]:
    if price_to_book > 0 and current_price > 0:
        beginning_bvps = current_price / price_to_book
    else:
        beginning_bvps = max(eps / 0.12, 1.0)

    residual_values = []
    book = beginning_bvps
    for _ in range(horizon):
        earnings = max(eps, 0.0)
        dividends = earnings * payout_ratio
        residual = earnings - (cost_of_equity * book)
        residual_values.append(residual)
        book = book + earnings - dividends

    pv_residual = _discount_cashflows(residual_values, cost_of_equity)
    terminal_residual = residual_values[-1] * (1.0 + terminal_growth_rate)
    terminal_value = terminal_residual / (cost_of_equity - terminal_growth_rate)
    intrinsic_per_share = beginning_bvps + pv_residual + (terminal_value / ((1.0 + cost_of_equity) ** horizon))
    return {
        "enterprise_value": intrinsic_per_share,
        "equity_value": intrinsic_per_share,
        "intrinsic_value_per_share": intrinsic_per_share,
    }


def _run_relative_bridge(
    *,
    revenue: float,
    shares_outstanding: float,
    peer_multiple_value: float,
    uncertainty_discount_pct: float,
    fcff_intrinsic_per_share: float,
    bridge_to_positive_fcff_year: float,
) -> dict[str, float]:
    revenue_per_share = revenue / shares_outstanding
    relative_raw = revenue_per_share * peer_multiple_value
    relative_after_discount = relative_raw * (1.0 - uncertainty_discount_pct)

    bridge_weight = _clamp((bridge_to_positive_fcff_year - 1.0) / 7.0, 0.1, 0.9)
    intrinsic_per_share = (bridge_weight * relative_after_discount) + ((1.0 - bridge_weight) * fcff_intrinsic_per_share)
    return {
        "enterprise_value": intrinsic_per_share,
        "equity_value": intrinsic_per_share,
        "intrinsic_value_per_share": intrinsic_per_share,
    }


def _build_wacc(
    assumptions: dict[str, float],
    market: dict[str, float],
    beta: float,
) -> tuple[float, float]:
    rf = _num(market.get("risk_free_rate"), 0.04)
    erp = _num(market.get("mature_erp"), 0.05)
    crp = _num(market.get("country_risk_premium"), 0.02)

    cost_of_equity = rf + (beta * erp) + crp
    cost_of_equity = max(cost_of_equity, assumptions["cost_of_equity"])

    debt_weight = assumptions["target_debt_weight"]
    equity_weight = 1.0 - debt_weight
    after_tax_debt = assumptions["cost_of_debt"] * (1.0 - assumptions["tax_rate"])
    wacc = (cost_of_equity * equity_weight) + (after_tax_debt * debt_weight)
    return _clamp(wacc, 0.04, 0.35), _clamp(cost_of_equity, 0.05, 0.45)


def _compute_fcff_base(
    *,
    assumptions: dict[str, float],
    metrics: dict[str, Any],
) -> dict[str, float]:
    revenue = max(_num(metrics.get("totalRevenue")), 0.0)
    shares = _num(metrics.get("sharesOutstanding"))
    if shares <= 0:
        raise ValueError("shares_outstanding must be > 0")

    horizon = 5
    growth_path = [assumptions["revenue_growth"]] * horizon
    margin_path = [assumptions["operating_margin"]] * horizon
    shares_future = shares * (1.0 - assumptions["shares_reduction_per_year"]) ** 5
    return _run_fcff(
        initial_revenue=revenue,
        revenue_growth_path=growth_path,
        operating_margin_path=margin_path,
        tax_rate=assumptions["tax_rate"],
        reinvestment_rate=assumptions["reinvestment_rate"],
        wacc=assumptions["wacc"],
        terminal_growth_rate=assumptions["terminal_growth_rate"],
        net_debt=assumptions["net_debt"],
        shares_outstanding=max(shares_future, 1.0),
    )


def _compute_fcff_two_stage(
    *,
    assumptions: dict[str, float],
    metrics: dict[str, Any],
) -> dict[str, float]:
    revenue = max(_num(metrics.get("totalRevenue")), 0.0)
    shares = _num(metrics.get("sharesOutstanding"))
    if shares <= 0:
        raise ValueError("shares_outstanding must be > 0")

    high_growth_years = int(round(assumptions["high_growth_years"]))
    fade_years = 5
    g0 = assumptions["revenue_growth"]
    g_term = assumptions["terminal_growth_rate"] + 0.005

    growth_path = [g0 for _ in range(high_growth_years)]
    for i in range(1, fade_years + 1):
        alpha = i / fade_years
        growth_path.append((1.0 - alpha) * g0 + (alpha * g_term))

    margin_start = assumptions["operating_margin"] * 0.9
    margin_end = assumptions["operating_margin"]
    margin_path = []
    for i in range(len(growth_path)):
        alpha = (i + 1) / len(growth_path)
        margin_path.append((1.0 - alpha) * margin_start + (alpha * margin_end))

    shares_future = shares * (1.0 - assumptions["shares_reduction_per_year"]) ** 5
    return _run_fcff(
        initial_revenue=revenue,
        revenue_growth_path=growth_path,
        operating_margin_path=margin_path,
        tax_rate=assumptions["tax_rate"],
        reinvestment_rate=assumptions["reinvestment_rate"],
        wacc=assumptions["wacc"],
        terminal_growth_rate=assumptions["terminal_growth_rate"],
        net_debt=assumptions["net_debt"],
        shares_outstanding=max(shares_future, 1.0),
    )


def _compute_fcff_normalized_cycle(
    *,
    assumptions: dict[str, float],
    metrics: dict[str, Any],
) -> dict[str, float]:
    revenue = max(_num(metrics.get("totalRevenue")), 0.0)
    shares = _num(metrics.get("sharesOutstanding"))
    if shares <= 0:
        raise ValueError("shares_outstanding must be > 0")

    normalized_revenue = max(_num(metrics.get("normalizedRevenue"), revenue * assumptions["cycle_normalization_factor"]), 0.0)
    normalized_margin = assumptions["normalized_operating_margin"]

    horizon = 7
    growth_path = []
    g = assumptions["revenue_growth"] * 0.8
    for _ in range(horizon):
        g = max(assumptions["terminal_growth_rate"], g - 0.01)
        growth_path.append(g)

    margin_path = [normalized_margin] * horizon
    shares_future = shares * (1.0 - assumptions["shares_reduction_per_year"]) ** 5
    return _run_fcff(
        initial_revenue=normalized_revenue,
        revenue_growth_path=growth_path,
        operating_margin_path=margin_path,
        tax_rate=assumptions["tax_rate"],
        reinvestment_rate=min(0.95, assumptions["reinvestment_rate"] + 0.05),
        wacc=assumptions["wacc"],
        terminal_growth_rate=assumptions["terminal_growth_rate"],
        net_debt=assumptions["net_debt"],
        shares_outstanding=max(shares_future, 1.0),
    )


def _compute_single_scenario(
    method: str,
    assumptions: dict[str, float],
    metrics: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    revenue = max(_num(metrics.get("totalRevenue")), 0.0)
    shares = _num(metrics.get("sharesOutstanding"))
    if shares <= 0:
        raise ValueError("shares_outstanding must be > 0")

    if method == "dividend_discount":
        ddm = _run_dividend_discount(
            eps=_num(metrics.get("earnings_per_share")),
            dividend_rate=_num(metrics.get("dividendRate")),
            payout_ratio=assumptions["payout_ratio"],
            earnings_growth=assumptions["earnings_growth"],
            cost_of_equity=assumptions["cost_of_equity"],
            terminal_growth_rate=assumptions["terminal_growth_rate"],
        )
        residual = _run_residual_income(
            current_price=_num(metrics.get("currentPrice")),
            price_to_book=_num(metrics.get("price_to_book_ratio")),
            eps=_num(metrics.get("earnings_per_share")),
            payout_ratio=assumptions["payout_ratio"],
            cost_of_equity=assumptions["cost_of_equity"],
            terminal_growth_rate=assumptions["terminal_growth_rate"],
        )
        blended = (ddm["intrinsic_value_per_share"] * 0.6) + (residual["intrinsic_value_per_share"] * 0.4)
        return (
            {
                "enterprise_value": blended,
                "equity_value": blended,
                "intrinsic_value_per_share": blended,
            },
            {
                "ddm_intrinsic_value_per_share": ddm["intrinsic_value_per_share"],
                "residual_income_intrinsic_value_per_share": residual["intrinsic_value_per_share"],
            },
        )

    if method == "residual_income":
        residual = _run_residual_income(
            current_price=_num(metrics.get("currentPrice")),
            price_to_book=_num(metrics.get("price_to_book_ratio")),
            eps=_num(metrics.get("earnings_per_share")),
            payout_ratio=assumptions["payout_ratio"],
            cost_of_equity=assumptions["cost_of_equity"],
            terminal_growth_rate=assumptions["terminal_growth_rate"],
        )
        return residual, {}

    if method == "fcff_dcf_two_stage":
        return _compute_fcff_two_stage(assumptions=assumptions, metrics=metrics), {}

    if method == "fcff_dcf_normalized_cycle":
        return _compute_fcff_normalized_cycle(assumptions=assumptions, metrics=metrics), {}

    if method == "relative_valuation_with_fcff_bridge":
        fcff_reference = _compute_fcff_two_stage(assumptions=assumptions, metrics=metrics)
        relative = _run_relative_bridge(
            revenue=revenue,
            shares_outstanding=shares,
            peer_multiple_value=assumptions["peer_multiple_value"],
            uncertainty_discount_pct=assumptions["uncertainty_discount_pct"],
            fcff_intrinsic_per_share=fcff_reference["intrinsic_value_per_share"],
            bridge_to_positive_fcff_year=assumptions["bridge_to_positive_fcff_year"],
        )
        return relative, {"fcff_bridge_intrinsic_value_per_share": fcff_reference["intrinsic_value_per_share"]}

    return _compute_fcff_base(assumptions=assumptions, metrics=metrics), {}


def _model_risk(method: str, asset_type: str) -> str:
    if method in {"relative_valuation_with_fcff_bridge"} or asset_type == "no-earnings":
        return "high"
    if method in {"fcff_dcf_two_stage", "fcff_dcf_normalized_cycle", "dividend_discount", "residual_income"}:
        return "medium"
    return "low"


def _data_quality(metrics: dict[str, Any], method: str) -> str:
    required = ["sharesOutstanding", "currentPrice"]
    if method in {"fcff_dcf", "fcff_dcf_two_stage", "fcff_dcf_normalized_cycle", "relative_valuation_with_fcff_bridge"}:
        required.extend(["totalRevenue", "profitMargins", "revenue_growth"])
    if method in {"dividend_discount", "residual_income"}:
        required.extend(["earnings_per_share", "price_to_book_ratio"])

    missing = 0
    for key in required:
        if metrics.get(key) in (None, ""):
            missing += 1
    ratio = missing / max(len(required), 1)
    if ratio > 0.4:
        return "low"
    if ratio > 0.15:
        return "medium"
    return "high"


def _assert_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite")


def _simulate_range(
    *,
    method: str,
    assumptions: dict[str, float],
    metrics: dict[str, Any],
    trials: int,
    seed: int,
) -> dict[str, float]:
    if trials <= 0:
        value = _compute_single_scenario(method, assumptions, metrics)[0]["intrinsic_value_per_share"]
        return {"p10": value, "p50": value, "p90": value}

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(trials):
        draw = dict(assumptions)
        draw["revenue_growth"] += rng.gauss(0.0, 0.012)
        draw["operating_margin"] += rng.gauss(0.0, 0.01)
        draw["wacc"] += rng.gauss(0.0, 0.008)
        draw["terminal_growth_rate"] += rng.gauss(0.0, 0.003)
        draw, _ = _sanitize_assumptions(draw, metrics.get("country") or "US", strict_mode_policy="autocorrect")
        value = _compute_single_scenario(method, draw, metrics)[0]["intrinsic_value_per_share"]
        if math.isfinite(value):
            samples.append(value)

    if not samples:
        base = _compute_single_scenario(method, assumptions, metrics)[0]["intrinsic_value_per_share"]
        return {"p10": base, "p50": base, "p90": base}

    samples.sort()

    def _pick(q: float) -> float:
        idx = int(round((len(samples) - 1) * q))
        return samples[max(0, min(idx, len(samples) - 1))]

    return {
        "p10": _pick(0.1),
        "p50": _pick(0.5),
        "p90": _pick(0.9),
    }


def run_valuation(
    *,
    ticker: str,
    metrics: dict[str, Any],
    asset_type: str,
    country: str = "US",
    margin_of_safety_pct: float = 0.25,
    llm_assumptions: dict[str, Any] | None = None,
    method_override: str | None = None,
    method_selection_reason: str | None = None,
    method_selected_by: str = "deterministic",
    method_selection_trace: list[dict[str, Any]] | None = None,
    valuation_date: str | None = None,
    assumption_source: str = "online",
    strict_mode: bool = True,
    strict_mode_policy: str | None = None,
    monte_carlo_trials: int = 1000,
    monte_carlo_seed: int = 42,
) -> dict[str, Any]:
    asset = (asset_type or "stable-cashflow").strip().lower()
    if asset not in ASSET_TYPES:
        asset = "stable-cashflow"

    policy = (strict_mode_policy or "autocorrect").strip().lower()
    if policy not in {"autocorrect", "raise"}:
        policy = "autocorrect"

    method = select_method(asset, has_dividends=_num(metrics.get("dividendRate")) > 0)
    if method_override and method_override in ALLOWED_METHODS:
        method = method_override

    market = get_market_assumptions(country=country, valuation_date=valuation_date)
    market_dict = market_assumptions_to_dict(market)

    beta = _num(metrics.get("beta"), 1.0)
    debt_to_equity = _clamp(_num(metrics.get("debt_to_equity"), 0.35), 0.0, 3.0)
    target_debt_weight = debt_to_equity / (1.0 + debt_to_equity)
    default_wacc_guess = _clamp(
        market.risk_free_rate + (beta * market.mature_erp) + market.country_risk_premium,
        0.06,
        0.25,
    )

    default_assumptions = {
        "revenue_growth": _num(metrics.get("revenue_growth"), 0.05),
        "earnings_growth": _num(metrics.get("earnings_growth"), _num(metrics.get("revenue_growth"), 0.04)),
        "operating_margin": _num(metrics.get("operating_margin"), _num(metrics.get("profitMargins"), 0.12)),
        "normalized_operating_margin": _num(
            metrics.get("normalized_operating_margin"),
            _num(metrics.get("profitMargins"), 0.12),
        ),
        "tax_rate": 0.21,
        "reinvestment_rate": 0.4,
        "wacc": default_wacc_guess,
        "terminal_growth_rate": _clamp(market.long_run_nominal_growth - 0.005, 0.015, 0.05),
        "net_debt": _num(metrics.get("totalDebt")) - _num(metrics.get("totalCash")),
        "shares_reduction_per_year": 0.005,
        "cost_of_equity": _clamp(default_wacc_guess + 0.005, 0.05, 0.4),
        "cost_of_debt": _clamp(_num(metrics.get("interest_rate"), market.risk_free_rate + 0.02), 0.01, 0.20),
        "target_debt_weight": target_debt_weight,
        "payout_ratio": 0.35,
        "peer_multiple_value": _num(metrics.get("price_to_sales_ratio"), 3.0),
        "uncertainty_discount_pct": 0.3,
        "high_growth_years": 5.0,
        "cycle_normalization_factor": 0.9,
        "bridge_to_positive_fcff_year": 3.0,
        "steady_state_roic": 0.10,
    }
    merged = _merge_assumptions(default_assumptions, llm_assumptions or {})
    assumptions, assumption_checks = _sanitize_assumptions(merged, country, strict_mode_policy=policy)

    computed_wacc, computed_ke = _build_wacc(assumptions, market_dict, beta)
    assumptions["wacc"] = max(assumptions["wacc"], computed_wacc)
    assumptions["cost_of_equity"] = max(assumptions["cost_of_equity"], computed_ke)
    assumptions, wacc_checks = _sanitize_assumptions(assumptions, country, strict_mode_policy=policy)

    mos_pct = _clamp(_num(margin_of_safety_pct, 0.25), 0.0, 0.8)
    scenarios: dict[str, Any] = {}
    check_results = list(assumption_checks) + list(wacc_checks)
    cross_checks: dict[str, Any] = {}
    for scenario_name in ("bear", "base", "bull"):
        scenario_assumptions, scenario_checks = _sanitize_assumptions(
            _scenario_assumptions(assumptions, scenario_name),
            country,
            strict_mode_policy=policy,
        )
        check_results.extend(
            {
                "name": f"{scenario_name}_{entry['name']}",
                "passed": entry["passed"],
                "severity": entry.get("severity", "warning"),
                "details": entry["details"],
            }
            for entry in scenario_checks
        )
        scenario_result, scenario_cross = _compute_single_scenario(method, scenario_assumptions, metrics)
        intrinsic = scenario_result["intrinsic_value_per_share"]
        _assert_finite(intrinsic, f"{scenario_name}.intrinsic_value_per_share")
        current_price = _num(metrics.get("currentPrice"))
        upside = ((intrinsic / current_price) - 1.0) * 100 if current_price > 0 else 0.0
        scenarios[scenario_name] = {
            "intrinsic_value_per_share": intrinsic,
            "upside_downside_pct": upside,
            "assumptions": scenario_assumptions,
            "enterprise_value": scenario_result["enterprise_value"],
            "equity_value": scenario_result["equity_value"],
        }
        if scenario_name == "base" and scenario_cross:
            cross_checks = scenario_cross

    valuation_range = _simulate_range(
        method=method,
        assumptions=scenarios["base"]["assumptions"],
        metrics=metrics,
        trials=max(0, int(monte_carlo_trials)),
        seed=int(monte_carlo_seed),
    )

    base_intrinsic = scenarios["base"]["intrinsic_value_per_share"]
    current_price = _num(metrics.get("currentPrice"))
    result = {
        "intrinsic_value_per_share": base_intrinsic,
        "current_price": current_price,
        "upside_downside_pct": ((base_intrinsic / current_price) - 1.0) * 100 if current_price > 0 else 0.0,
        "margin_of_safety_price": base_intrinsic * (1.0 - mos_pct),
        "valuation_range": valuation_range,
    }
    _assert_finite(result["intrinsic_value_per_share"], "result.intrinsic_value_per_share")

    critical_fails = [
        check for check in check_results if not check.get("passed", True) and check.get("severity") == "critical"
    ]
    diagnostics_note = "ok"
    if policy == "autocorrect" and critical_fails:
        diagnostics_note = "strict checks autocorrected"
    if strict_mode and policy == "raise" and critical_fails:
        diagnostics_note = "strict raise policy enabled"

    diagnostics = {
        "model_risk": _model_risk(method, asset),
        "data_quality": _data_quality(metrics, method),
        "checks": check_results,
        "critical_failures": critical_fails,
        "assumption_freshness": {
            "source": market.source,
            "as_of_date": market.as_of_date,
            "staleness_days": market.staleness_days,
        },
        "note": diagnostics_note,
    }

    output = {
        "ticker": ticker.upper().strip(),
        "valuation_method": method,
        "method_selection": {
            "selected_by": method_selected_by,
            "asset_type": asset,
            "reason": method_selection_reason or f"Method mapped by deterministic asset profile rule ({asset}).",
            "trace": method_selection_trace or [],
        },
        "valuation_metadata": {
            "valuation_date": valuation_date,
            "assumption_source": assumption_source,
            "strict_mode_policy": policy,
            "monte_carlo_trials": int(max(0, monte_carlo_trials)),
            "market_assumptions": market_dict,
        },
        "assumptions": {
            "operating": {
                "revenue_growth": assumptions["revenue_growth"],
                "earnings_growth": assumptions["earnings_growth"],
                "operating_margin": assumptions["operating_margin"],
                "normalized_operating_margin": assumptions["normalized_operating_margin"],
                "tax_rate": assumptions["tax_rate"],
                "reinvestment_rate": assumptions["reinvestment_rate"],
                "shares_reduction_per_year": assumptions["shares_reduction_per_year"],
                "steady_state_roic": assumptions["steady_state_roic"],
            },
            "financing": {
                "wacc": assumptions["wacc"],
                "cost_of_equity": assumptions["cost_of_equity"],
                "cost_of_debt": assumptions["cost_of_debt"],
                "target_debt_weight": assumptions["target_debt_weight"],
                "net_debt": assumptions["net_debt"],
            },
            "terminal": {
                "terminal_growth_rate": assumptions["terminal_growth_rate"],
            },
            "relative": {
                "peer_multiple_value": assumptions["peer_multiple_value"],
                "uncertainty_discount_pct": assumptions["uncertainty_discount_pct"],
                "bridge_to_positive_fcff_year": assumptions["bridge_to_positive_fcff_year"],
            },
            "method_specific": {
                "high_growth_years": assumptions["high_growth_years"],
                "cycle_normalization_factor": assumptions["cycle_normalization_factor"],
                "payout_ratio": assumptions["payout_ratio"],
            },
        },
        "scenarios": scenarios,
        "result": result,
        "diagnostics": diagnostics,
        "cross_checks": cross_checks,
        "rationale": (
            f"Method selected: `{method}`. "
            f"Selection reason: {method_selection_reason or f'asset profile {asset} mapping'}. "
            "Valuation computed with deterministic engine plus Monte Carlo range."
        ),
        # Compatibility keys for old consumers
        "intrinsic_value_per_share": round(result["intrinsic_value_per_share"], 2),
        "current_price": current_price,
        "undervalued": result["margin_of_safety_price"] >= current_price if current_price > 0 else False,
    }
    return output
