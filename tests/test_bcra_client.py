from __future__ import annotations

from scripts import bcra


def test_fetch_bcra_data_uses_v4_and_parses_nested_detail(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyResponse:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict:
            return {
                "status": 200,
                "results": [
                    {
                        "idVariable": 4,
                        "detalle": [
                            {"fecha": "2026-08-19", "valor": 1517.54},
                            {"fecha": "2026-08-18", "valor": "1515.02"},
                        ],
                    }
                ],
            }

    def fake_get(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return DummyResponse()

    monkeypatch.setattr(bcra.requests, "get", fake_get)

    result = bcra.fetch_bcra_data(4, "2026-08-18", "2026-08-19")

    assert captured["url"] == "https://api.bcra.gob.ar/estadisticas/v4.0/monetarias/4"
    assert captured["verify"] is True
    assert captured["params"] == {"desde": "2026-08-18", "hasta": "2026-08-19", "limit": 1000}
    assert result == [
        {"fecha": bcra.datetime(2026, 8, 18).date(), "valor": 1515.02},
        {"fecha": bcra.datetime(2026, 8, 19).date(), "valor": 1517.54},
    ]
