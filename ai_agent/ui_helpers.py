from __future__ import annotations

import requests
from urllib.parse import quote_plus, urlparse


def build_finviz_chart_url(ticker: str) -> str:
    encoded = quote_plus((ticker or "").strip().upper())
    return (
        "https://charts-node.finviz.com/chart.ashx"
        f"?cs=l&t={encoded}&tf=d&s=linear&ct=candle_stick&tm=l"
        "&o[0][ot]=sma&o[0][op]=50&o[0][oc]=FF8F33C6"
        "&o[1][ot]=sma&o[1][op]=200&o[1][oc]=DCB3326D"
        "&o[2][ot]=sma&o[2][op]=20&o[2][oc]=DC32B363"
        "&o[3][ot]=patterns&o[3][op]=&o[3][oc]=000"
    )


def _extract_domain(website: str | None) -> str | None:
    if not website:
        return None
    raw = website.strip()
    if not raw:
        return None

    if "://" not in raw:
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    domain = (parsed.netloc or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]
    if not domain or "." not in domain:
        return None

    return domain


def build_logo_candidates(website: str | None) -> list[str]:
    domain = _extract_domain(website)
    if not domain:
        return []
    return [
        f"https://logo.clearbit.com/{domain}",
        f"https://icons.duckduckgo.com/ip3/{domain}.ico",
        f"https://www.google.com/s2/favicons?domain={domain}&sz=128",
    ]


def resolve_logo_url(website: str | None, timeout: int = 4) -> str | None:
    candidates = build_logo_candidates(website)
    for url in candidates:
        try:
            response = requests.get(url, timeout=timeout)
            content_type = (response.headers.get("content-type") or "").lower()
            if response.status_code == 200 and "image" in content_type:
                return url
        except Exception:
            continue
    return candidates[0] if candidates else None


def build_logo_url(website: str | None) -> str | None:
    # Backward-compatible alias for older imports.
    return resolve_logo_url(website)


def extract_summary_fields(parsed_decision: dict) -> dict:
    if not isinstance(parsed_decision, dict):
        parsed_decision = {"raw_output": parsed_decision}

    agent_signals = parsed_decision.get("agent_signals")
    if not isinstance(agent_signals, list):
        agent_signals = []

    return {
        "action": str(parsed_decision.get("action", "N/A")).upper(),
        "confidence": str(parsed_decision.get("confidence", "N/A")),
        "price": parsed_decision.get("price"),
        "amount": parsed_decision.get("amount"),
        "quantity": parsed_decision.get("quantity"),
        "reasoning": parsed_decision.get("reasoning", "No reasoning provided."),
        "news": parsed_decision.get("news"),
        "agent_signals": agent_signals,
    }
