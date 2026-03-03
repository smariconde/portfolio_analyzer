import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ai_agent.env_loader import load_project_env
from ai_agent.tools import get_financial_metrics, get_news
from ai_agent.valuation_v2.assistant import (
    deterministic_decision_summary,
    deterministic_method_fallback,
    propose_decision_summary_with_llm,
    propose_method_and_assumptions_with_llm,
)
from ai_agent.valuation_v2.engine import run_valuation
from ai_agent.valuation_v2.schemas import validate_output_schema

load_project_env()

DEFAULT_MARGIN_OF_SAFETY_PCT = 0.25
DEFAULT_STRICT_MODE = True
DEFAULT_ASSET_TYPE_FALLBACK = "stable-cashflow"


def build_sensitivity_table(payload: dict[str, Any], ticker: str, metrics: dict[str, Any]) -> pd.DataFrame:
    base = payload["assumptions"]["financing"]["wacc"]
    term = payload["assumptions"]["terminal"]["terminal_growth_rate"]
    base_assumptions = payload["scenarios"]["base"]["assumptions"]
    method_selection = payload.get("method_selection", {})
    valuation_meta = payload.get("valuation_metadata", {})
    rows = []
    for w_shift in (-0.01, -0.005, 0.0, 0.005, 0.01):
        for g_shift in (-0.005, 0.0, 0.005):
            llm_assumptions = dict(base_assumptions)
            llm_assumptions["wacc"] = base + w_shift
            llm_assumptions["terminal_growth_rate"] = term + g_shift
            recalculated = run_valuation(
                ticker=ticker,
                metrics=metrics,
                asset_type=method_selection.get("asset_type", DEFAULT_ASSET_TYPE_FALLBACK),
                country=(metrics.get("country") or "US").upper().strip(),
                margin_of_safety_pct=DEFAULT_MARGIN_OF_SAFETY_PCT,
                llm_assumptions=llm_assumptions,
                method_override=payload.get("valuation_method"),
                method_selection_reason=method_selection.get("reason"),
                method_selected_by=method_selection.get("selected_by", "deterministic_fallback"),
                method_selection_trace=method_selection.get("trace", []),
                valuation_date=valuation_meta.get("valuation_date"),
                assumption_source=valuation_meta.get("assumption_source", "online"),
                strict_mode=DEFAULT_STRICT_MODE,
                strict_mode_policy=valuation_meta.get("strict_mode_policy", "autocorrect"),
                monte_carlo_trials=0,
            )
            rows.append(
                {
                    "wacc": round(llm_assumptions["wacc"], 4),
                    "terminal_growth": round(llm_assumptions["terminal_growth_rate"], 4),
                    "intrinsic_value_per_share": round(recalculated["result"]["intrinsic_value_per_share"], 2),
                }
            )
    return pd.DataFrame(rows)


@st.cache_data(ttl=21600)
def load_valuation_v2_cached(
    ticker: str,
) -> tuple[dict[str, Any], str | None, list[str], dict[str, Any], dict[str, Any]]:
    metrics = get_financial_metrics(ticker=ticker)
    country = (metrics.get("country") or "US").upper().strip()
    selection = deterministic_method_fallback(
        ticker=ticker,
        metrics=metrics,
        country=country,
        asset_type="",
    )

    if os.environ.get("GOOGLE_API_KEY"):
        try:
            selection = propose_method_and_assumptions_with_llm(
                ticker=ticker,
                metrics=metrics,
                country=country,
            )
        except Exception:
            selection = deterministic_method_fallback(
                ticker=ticker,
                metrics=metrics,
                country=country,
                asset_type="",
            )

    valuation = run_valuation(
        ticker=ticker,
        metrics=metrics,
        asset_type=selection["asset_type"],
        country=country,
        margin_of_safety_pct=DEFAULT_MARGIN_OF_SAFETY_PCT,
        llm_assumptions=selection.get("assumptions", {}),
        method_override=selection.get("valuation_method"),
        method_selection_reason=selection.get("valuation_method_reason"),
        method_selected_by=selection.get("selected_by", "deterministic_fallback"),
        method_selection_trace=selection.get("method_selection_trace", []),
        assumption_source="online",
        strict_mode=DEFAULT_STRICT_MODE,
        strict_mode_policy="autocorrect",
        monte_carlo_trials=1500,
    )

    valuation["metadata_metrics"] = metrics

    news_payload: dict[str, Any] = {"answer": "", "results": []}
    if os.environ.get("TAVILY_API_KEY"):
        try:
            news_payload = get_news(
                query=f"Ultimas noticias financieras y operativas de {ticker}",
                days=30,
                max_results=5,
            )
        except Exception:
            news_payload = {"answer": "", "results": []}
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            decision_payload = propose_decision_summary_with_llm(
                ticker=ticker,
                metrics=metrics,
                valuation_payload=valuation,
                news_payload=news_payload,
            )
        except Exception:
            decision_payload = deterministic_decision_summary(
                ticker=ticker,
                metrics=metrics,
                valuation_payload=valuation,
                news_payload=news_payload,
            )
    else:
        decision_payload = deterministic_decision_summary(
            ticker=ticker,
            metrics=metrics,
            valuation_payload=valuation,
            news_payload=news_payload,
        )

    valuation["decision_summary"] = decision_payload.get("decision_summary", {})
    return valuation, metrics.get("logo_url"), metrics.get("logo_candidates", []), news_payload, decision_payload


if __name__ in {"__main__", "__page__"}:
    st.title("Valuation Agent V2.0 :material/monitoring:")
    st.caption("Automatic method selection + deterministic valuation math.")
    st.info(
        "Enter a ticker and run. The agent selects the method automatically and explains why.",
        icon=":material/info:",
    )

    if st.button("Refresh valuation cache", icon=":material/refresh:"):
        st.cache_data.clear()
        st.rerun()

    with st.form("valuation_v2_form"):
        ticker = st.text_input("Ticker", "AAPL", max_chars=8).upper().strip()
        submit = st.form_submit_button("Run Valuation V2", type="primary")

    if submit:
        if not ticker:
            st.error("Enter a valid ticker.")
            st.stop()

        with st.spinner("Running deterministic valuation engine..."):
            try:
                payload, logo, logo_candidates, news_payload, decision_payload = load_valuation_v2_cached(
                    ticker=ticker,
                )
            except Exception as exc:
                st.error(f"Valuation V2 failed: {exc}")
                st.stop()

        valid, schema_errors = validate_output_schema(payload)
        if not valid:
            st.error("Output schema validation failed.")
            st.write(schema_errors)
            st.stop()

        st.divider()
        h1, h2 = st.columns([1, 9], gap="small", vertical_alignment="center")
        with h1:
            rendered_logo = False
            for candidate in [logo, *logo_candidates]:
                if not candidate:
                    continue
                try:
                    st.image(candidate, width=100)
                    rendered_logo = True
                    break
                except Exception:
                    continue
            if not rendered_logo:
                st.caption("Logo no disponible")
        with h2:
            st.markdown(f"**[{ticker}](https://finviz.com/quote.ashx?t={ticker}&p=d)**")
            st.caption(f"Method: `{payload['valuation_method']}`")
            selection_meta = payload.get("method_selection", {})
            st.caption(
                f"Selected by: `{selection_meta.get('selected_by', 'unknown')}` | "
                f"Asset type: `{selection_meta.get('asset_type', 'n/a')}`"
            )
            st.markdown(f"**Why this method:** {selection_meta.get('reason', 'No reason available')}")
            st.caption("Defaults: margin_of_safety=25%, strict_mode=on, country inferred from company profile.")
            freshness = payload.get("diagnostics", {}).get("assumption_freshness", {})
            st.caption(
                f"Market assumptions: `{freshness.get('source', 'unknown')}` | "
                f"as_of `{freshness.get('as_of_date', 'n/a')}` | "
                f"staleness {freshness.get('staleness_days', 'n/a')}d"
            )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Intrinsic (Base)", f"${payload['result']['intrinsic_value_per_share']:.2f}")
        with k2:
            st.metric("Current Price", f"${payload['result']['current_price']:.2f}")
        with k3:
            st.metric("Upside/Downside", f"{payload['result']['upside_downside_pct']:.2f}%")
        with k4:
            st.metric("MoS Price", f"${payload['result']['margin_of_safety_price']:.2f}")
        vr = payload["result"].get("valuation_range", {})
        if vr:
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("P10", f"${vr.get('p10', 0.0):.2f}")
            with r2:
                st.metric("P50", f"${vr.get('p50', 0.0):.2f}")
            with r3:
                st.metric("P90", f"${vr.get('p90', 0.0):.2f}")

        t1, t2, t3, t4, t5, t6 = st.tabs(
            ["Scenarios", "Assumptions", "Diagnostics", "Decision", "News Context", "Raw JSON"]
        )
        with t1:
            scenario_rows = []
            for name in ("bear", "base", "bull"):
                data = payload["scenarios"][name]
                scenario_rows.append(
                    {
                        "scenario": name,
                        "intrinsic_value_per_share": round(data["intrinsic_value_per_share"], 2),
                        "upside_downside_pct": round(data["upside_downside_pct"], 2),
                        "enterprise_value": round(data["enterprise_value"], 2),
                        "equity_value": round(data["equity_value"], 2),
                    }
                )
            st.dataframe(pd.DataFrame(scenario_rows), width="stretch", hide_index=True)
            st.markdown("**Sensitivity (recalculated): WACC x Terminal Growth**")
            st.dataframe(build_sensitivity_table(payload, ticker, payload.get("metadata_metrics", {})), width="stretch", hide_index=True)

        with t2:
            st.json(payload["assumptions"])

        with t3:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Model Risk", payload["diagnostics"]["model_risk"])
            with c2:
                st.metric("Data Quality", payload["diagnostics"]["data_quality"])
            st.write(payload["diagnostics"].get("note", "ok"))
            checks_df = pd.DataFrame(payload["diagnostics"]["checks"])
            st.dataframe(checks_df, width="stretch", hide_index=True)

        with t4:
            summary = decision_payload.get("decision_summary", {})
            st.markdown("**Contexto de mercado**")
            st.write(summary.get("market_context", "Sin contexto disponible"))
            st.markdown("**Decision base**")
            st.write(summary.get("base_case_decision", "Sin decision disponible"))
            st.markdown("**Playbook por escenario**")
            playbook = summary.get("scenario_playbook", {})
            st.write(
                {
                    "bear": playbook.get("bear", ""),
                    "base": playbook.get("base", ""),
                    "bull": playbook.get("bull", ""),
                }
            )
            st.markdown("**Riesgos clave**")
            st.write(summary.get("key_risks", []))
            st.markdown("**Senales a monitorear**")
            st.write(summary.get("signals_to_monitor", []))
            if summary.get("model_used"):
                st.caption(f"Decision summary model: `{summary.get('model_used')}`")

        with t5:
            if not os.environ.get("TAVILY_API_KEY"):
                st.info("Configura `TAVILY_API_KEY` para ver contexto de noticias.")
            elif not news_payload.get("answer") and not news_payload.get("results"):
                st.warning("No se pudieron recuperar noticias para este ticker.")
            else:
                if news_payload.get("answer"):
                    st.markdown("**Resumen de contexto (IA):**")
                    st.write(news_payload["answer"])
                if news_payload.get("results"):
                    rows = []
                    for item in news_payload["results"]:
                        rows.append(
                            {
                                "title": item.get("title", ""),
                                "source": item.get("source", ""),
                                "published_date": item.get("published_date", ""),
                                "url": item.get("url", ""),
                            }
                        )
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        with t6:
            st.json(payload)
