from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ai_agent.llm_utils import invoke_gemini
from ai_agent.tools import parse_output_to_json
from ai_agent.valuation_v2.engine import ASSET_TYPES, METHOD_BY_ASSET_TYPE, select_method


ALLOWED_METHODS = {
    "dividend_discount",
    "fcff_dcf",
    "fcff_dcf_two_stage",
    "fcff_dcf_normalized_cycle",
    "relative_valuation_with_fcff_bridge",
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_asset_type_from_metrics(metrics: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    sector = str(metrics.get("sector", "")).strip().lower()
    industry = str(metrics.get("industry", "")).strip().lower()
    revenue_growth = _to_float(metrics.get("revenue_growth"))
    profit_margins = _to_float(metrics.get("profitMargins"))
    eps = _to_float(metrics.get("earnings_per_share"))
    dividend = _to_float(metrics.get("dividendRate"))

    trace: list[dict[str, Any]] = []

    if any(tag in sector for tag in ("financial", "bank", "insurance")):
        trace.append({"rule": "sector_financial", "matched": True, "weight": 1.0})
        return "financial", trace
    if "bank" in industry or "insurance" in industry:
        trace.append({"rule": "industry_financial", "matched": True, "weight": 1.0})
        return "financial", trace

    if eps <= 0 and profit_margins <= 0:
        trace.append({"rule": "negative_earnings_profile", "matched": True, "weight": 0.95})
        return "no-earnings", trace

    if any(tag in sector for tag in ("energy", "materials")):
        trace.append({"rule": "commodity_sector", "matched": True, "weight": 0.85})
        return "commodity", trace

    if any(tag in sector for tag in ("industrials", "consumer cyclical", "real estate")):
        trace.append({"rule": "cyclical_sector", "matched": True, "weight": 0.8})
        return "cyclical", trace

    if revenue_growth >= 0.15 and dividend <= 0:
        trace.append({"rule": "high_growth_signal", "matched": True, "weight": 0.75})
        return "high-growth", trace

    trace.append({"rule": "default_stable_cashflow", "matched": True, "weight": 0.6})
    return "stable-cashflow", trace


def propose_method_and_assumptions_with_llm(
    *,
    ticker: str,
    metrics: dict[str, Any],
    country: str,
) -> dict[str, Any]:
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a valuation method selector and assumptions assistant.
                Select the best valuation method for the specific company and output JSON only.
                Allowed asset_type: financial, stable-cashflow, high-growth, cyclical, commodity, no-earnings.
                Allowed valuation_method: dividend_discount, fcff_dcf, fcff_dcf_two_stage, fcff_dcf_normalized_cycle, relative_valuation_with_fcff_bridge.
                Include concise rationale based on business profile and accounting constraints.
                Write valuation_method_reason in Spanish (Argentina), using clear rioplatense wording.
                Return keys:
                asset_type, valuation_method, valuation_method_reason, confidence_pct,
                assumptions (object with numeric keys):
                revenue_growth, operating_margin, tax_rate, reinvestment_rate, wacc,
                terminal_growth_rate, net_debt, shares_reduction_per_year,
                cost_of_equity, payout_ratio, peer_multiple_value, uncertainty_discount_pct, high_growth_years.
                """,
            ),
            (
                "human",
                f"""Ticker: {ticker}
                Country: {country}
                Sector: {metrics.get("sector")}
                Industry: {metrics.get("industry")}
                Current price: {metrics.get("currentPrice")}
                Revenue (LTM): {metrics.get("totalRevenue")}
                Revenue growth: {metrics.get("revenue_growth")}
                Operating margin: {metrics.get("operating_margin")}
                Profit margin: {metrics.get("profitMargins")}
                Shares outstanding: {metrics.get("sharesOutstanding")}
                EPS: {metrics.get("earnings_per_share")}
                Dividend rate: {metrics.get("dividendRate")}
                Beta: {metrics.get("beta")}
                Total debt: {metrics.get("totalDebt")}
                Total cash: {metrics.get("totalCash")}
                Price/Sales: {metrics.get("price_to_sales_ratio")}
                Return only JSON.
                """,
            ),
        ]
    )
    prompt = template.invoke({})
    result, model_used = invoke_gemini(prompt, temperature=0.0, max_tokens=None, max_retries=4, stop=None)
    try:
        payload = parse_output_to_json(result.content)
    except json.JSONDecodeError:
        payload = {}
    parsed = _safe_dict(payload)

    inferred_asset, trace = infer_asset_type_from_metrics(metrics)
    asset_type = str(parsed.get("asset_type", "")).strip().lower()
    if asset_type not in ASSET_TYPES:
        asset_type = inferred_asset
    valuation_method = str(parsed.get("valuation_method", "")).strip().lower()
    if valuation_method not in ALLOWED_METHODS:
        valuation_method = select_method(
            asset_type, has_dividends=(metrics.get("dividendRate") or 0) > 0
        )

    return {
        "selected_by": "llm",
        "model_used": model_used,
        "asset_type": asset_type,
        "valuation_method": valuation_method,
        "valuation_method_reason": str(
            parsed.get(
                "valuation_method_reason",
                f"LLM fallback reason unavailable. Method mapped from asset_type={asset_type}.",
            )
        ),
        "confidence_pct": float(parsed.get("confidence_pct", 65.0) or 65.0),
        "method_selection_trace": trace,
        "assumptions": _safe_dict(parsed.get("assumptions")),
    }


def deterministic_method_fallback(
    *,
    ticker: str,
    metrics: dict[str, Any],
    country: str,
    asset_type: str = "stable-cashflow",
) -> dict[str, Any]:
    inferred_asset, trace = infer_asset_type_from_metrics(metrics)
    normalized_asset = (asset_type or inferred_asset).strip().lower()
    if normalized_asset not in ASSET_TYPES:
        normalized_asset = inferred_asset
    method = select_method(normalized_asset, has_dividends=(metrics.get("dividendRate") or 0) > 0)
    canonical_map = {v: k for k, v in METHOD_BY_ASSET_TYPE.items()}
    inferred_from_method = canonical_map.get(method, normalized_asset)
    return {
        "selected_by": "deterministic_fallback",
        "model_used": None,
        "asset_type": inferred_from_method,
        "valuation_method": method,
        "valuation_method_reason": (
            f"Fallback deterministic mapping for {ticker} using asset_type={inferred_from_method}, "
            f"country={country}."
        ),
        "confidence_pct": 55.0,
        "method_selection_trace": trace,
        "assumptions": {},
    }


def propose_decision_summary_with_llm(
    *,
    ticker: str,
    metrics: dict[str, Any],
    valuation_payload: dict[str, Any],
    news_payload: dict[str, Any],
) -> dict[str, Any]:
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Sos un analista fundamental y de valuacion.
                Con base en metricas, metodo de valuacion, escenarios y noticias, devolve SOLO JSON en espanol argentino.
                Estructura requerida:
                {
                  "decision_summary": {
                    "market_context": "<string>",
                    "base_case_decision": "<string>",
                    "scenario_playbook": {
                      "bear": "<string>",
                      "base": "<string>",
                      "bull": "<string>"
                    },
                    "key_risks": ["<string>", "<string>", "<string>"],
                    "signals_to_monitor": ["<string>", "<string>", "<string>"]
                  }
                }
                Tono: directo, sin jerga vacia, accionable.
                """,
            ),
            (
                "human",
                f"""Ticker: {ticker}
                Sector: {metrics.get("sector")}
                Industry: {metrics.get("industry")}
                Country: {metrics.get("country")}
                Core metrics:
                  - currentPrice: {metrics.get("currentPrice")}
                  - revenueGrowth: {metrics.get("revenue_growth")}
                  - earningsGrowth: {metrics.get("earnings_growth")}
                  - operatingMargin: {metrics.get("operating_margin")}
                  - profitMargins: {metrics.get("profitMargins")}
                  - debtToEquity: {metrics.get("debt_to_equity")}
                  - beta: {metrics.get("beta")}
                Valuation payload: {valuation_payload}
                News summary: {news_payload.get("answer", "")}
                News links: {news_payload.get("results", [])}
                """,
            ),
        ]
    )
    prompt = template.invoke({})
    result, model_used = invoke_gemini(prompt, temperature=0.1, max_tokens=None, max_retries=4, stop=None)
    try:
        payload = parse_output_to_json(result.content)
    except json.JSONDecodeError:
        payload = {}
    parsed = _safe_dict(payload)
    summary = _safe_dict(parsed.get("decision_summary"))
    if not summary:
        return deterministic_decision_summary(
            ticker=ticker,
            metrics=metrics,
            valuation_payload=valuation_payload,
            news_payload=news_payload,
        )
    summary["model_used"] = model_used
    return {"decision_summary": summary}


def deterministic_decision_summary(
    *,
    ticker: str,
    metrics: dict[str, Any],
    valuation_payload: dict[str, Any],
    news_payload: dict[str, Any],
) -> dict[str, Any]:
    scenarios = valuation_payload.get("scenarios", {})
    base_upside = _safe_dict(valuation_payload.get("result")).get("upside_downside_pct", 0.0)
    method = valuation_payload.get("valuation_method", "unknown")
    market_context = (
        f"{ticker} opera en {metrics.get('sector', 'sector no disponible')} / "
        f"{metrics.get('industry', 'industria no disponible')}. "
        f"Metodo aplicado: {method}."
    )
    return {
        "decision_summary": {
            "market_context": market_context,
            "base_case_decision": (
                "Sesgo comprador gradual" if base_upside > 15 else
                "Mantener y esperar confirmaciones" if base_upside > -10 else
                "Sesgo defensivo y preservacion de capital"
            ),
            "scenario_playbook": {
                "bear": (
                    f"Si se materializa bear ({_safe_dict(scenarios.get('bear')).get('upside_downside_pct', 0.0):.1f}%), "
                    "priorizar control de riesgo y reducir exposicion."
                ),
                "base": (
                    f"En base ({_safe_dict(scenarios.get('base')).get('upside_downside_pct', 0.0):.1f}%), "
                    "mantener disciplina de entrada/salida y monitorear ejecucion del negocio."
                ),
                "bull": (
                    f"En bull ({_safe_dict(scenarios.get('bull')).get('upside_downside_pct', 0.0):.1f}%), "
                    "habilita incrementar posicion por tramos."
                ),
            },
            "key_risks": [
                "Compresion de margenes o menor crecimiento al proyectado.",
                "Riesgo macro/tasa que eleve el costo de capital.",
                "Noticias negativas de ejecucion o guidance.",
            ],
            "signals_to_monitor": [
                "Revision de guidance y resultados trimestrales.",
                "Evolucion de margenes operativos y deuda neta.",
                "Cambios en beta/volatilidad y multiples del sector.",
            ],
            "model_used": "deterministic_fallback",
            "news_snapshot": news_payload.get("answer", ""),
        }
    }
