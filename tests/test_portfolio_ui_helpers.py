from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_agent.ui_helpers import build_finviz_chart_url, build_logo_url, extract_summary_fields


class _FakeResponse:
    def __init__(self, status_code: int, content_type: str = "") -> None:
        self.status_code = status_code
        self.headers = {"content-type": content_type}


def _unavailable_logo(*args, **kwargs) -> _FakeResponse:
    return _FakeResponse(status_code=404)


def test_build_finviz_chart_url_normalizes_ticker() -> None:
    url = build_finviz_chart_url(" aapl ")
    assert "t=AAPL" in url
    assert "charts-node.finviz.com" in url


def test_build_finviz_chart_url_encodes_special_chars() -> None:
    url = build_finviz_chart_url("BRK.B")
    assert "t=BRK.B" in url


def test_build_logo_url_from_https(monkeypatch) -> None:
    monkeypatch.setattr("ai_agent.ui_helpers.requests.get", _unavailable_logo)
    assert build_logo_url("https://www.microsoft.com/en-us") == "https://logo.clearbit.com/microsoft.com"


def test_build_logo_url_from_plain_domain(monkeypatch) -> None:
    monkeypatch.setattr("ai_agent.ui_helpers.requests.get", _unavailable_logo)
    assert build_logo_url("apple.com") == "https://logo.clearbit.com/apple.com"


def test_build_logo_url_uses_first_available_provider(monkeypatch) -> None:
    def fake_get(url: str, timeout: int) -> _FakeResponse:
        if "duckduckgo.com" in url:
            return _FakeResponse(status_code=200, content_type="image/png")
        return _FakeResponse(status_code=404)

    monkeypatch.setattr("ai_agent.ui_helpers.requests.get", fake_get)

    assert build_logo_url("apple.com") == "https://icons.duckduckgo.com/ip3/apple.com.ico"


def test_build_logo_url_invalid_values() -> None:
    assert build_logo_url(None) is None
    assert build_logo_url("") is None
    assert build_logo_url("invalid-domain") is None


def test_extract_summary_fields_complete_payload() -> None:
    parsed = {
        "action": "buy",
        "confidence": "72%",
        "price": 105.5,
        "amount": 1200,
        "quantity": 11,
        "reasoning": "Signals are aligned.",
        "news": "No material events.",
        "agent_signals": [{"agent": "quant", "signal": "bullish", "confidence": "70%"}],
    }
    summary = extract_summary_fields(parsed)
    assert summary["action"] == "BUY"
    assert summary["confidence"] == "72%"
    assert summary["agent_signals"][0]["agent"] == "quant"


def test_extract_summary_fields_partial_payload() -> None:
    summary = extract_summary_fields({"action": "hold"})
    assert summary["action"] == "HOLD"
    assert summary["reasoning"] == "No reasoning provided."
    assert summary["agent_signals"] == []


def test_extract_summary_fields_non_dict_payload() -> None:
    summary = extract_summary_fields("raw output")
    assert summary["action"] == "N/A"
    assert summary["agent_signals"] == []
