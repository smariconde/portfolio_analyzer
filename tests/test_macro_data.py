from __future__ import annotations

import pandas as pd

from portfolio_analyzer.data import macro


def test_month_to_number_known_values() -> None:
    assert macro._month_to_number("enero") == 1
    assert macro._month_to_number("sep") == 9
    assert macro._month_to_number("DICIEMBRE") == 12
    assert macro._month_to_number("Octubre*") == 10
    assert macro._month_to_number("Mes 11") == 11


def test_records_to_df_sorts_and_sets_datetime_index() -> None:
    records = [
        {"fecha": "2025-01-02", "valor": "2"},
        {"fecha": "2025-01-01", "valor": "1"},
    ]
    df = macro._records_to_df(records)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2025-01-01", "2025-01-02"]
    assert list(df["valor"]) == [1.0, 2.0]


def test_get_bcra_series_applies_adjustment(monkeypatch) -> None:
    monkeypatch.setattr(
        macro,
        "fetch_bcra_data",
        lambda variable_id, fecha_desde, fecha_hasta, limit: [
            {"fecha": pd.Timestamp("2025-01-01").date(), "valor": 100.0},
            {"fecha": pd.Timestamp("2025-01-02").date(), "valor": 200.0},
        ],
    )
    monkeypatch.setattr(
        macro,
        "get_minorista_exchange_rate",
        lambda fecha_desde, fecha_hasta: [
            {"fecha": pd.Timestamp("2025-01-01").date(), "valor": 10.0},
            {"fecha": pd.Timestamp("2025-01-02").date(), "valor": 20.0},
        ],
    )

    result = macro.get_bcra_series(variable_id=4, adjustment="usd")
    assert result.variable_id == 4
    assert list(result.dataframe["valor"]) == [10.0, 10.0]


def test_parse_indec_dataframe_from_period_and_month_columns() -> None:
    raw = pd.DataFrame(
        {
            "Periodo": [2024, None, 2025],
            "Mes": ["enero", "febrero", "ene"],
            "Exportaciones Total mensual": ["1.000,5", "1.200,5", "1300,5"],
            "Importaciones Total mensual": ["900,5", "1000,5", "1100,5"],
            "Saldo": ["100,0", "200,0", "200,0"],
        }
    )
    parsed = macro._parse_indec_dataframe(raw)
    assert not parsed.empty
    assert list(parsed.index.strftime("%Y-%m-%d"))[0] == "2024-01-01"
    assert float(parsed.iloc[0]["Exportaciones"]) == 1000.5
    assert float(parsed.iloc[0]["Importaciones"]) == 900.5


def test_parse_indec_dataframe_from_fecha_column() -> None:
    raw = pd.DataFrame(
        {
            "Fecha": ["01/01/2025", "01/02/2025"],
            "Exportaciones": [1500, 1600],
            "Importaciones": [1200, 1300],
        }
    )
    parsed = macro._parse_indec_dataframe(raw)
    assert not parsed.empty
    assert list(parsed.columns) == ["Exportaciones", "Importaciones", "Balanza Comercial"]
    assert float(parsed.iloc[0]["Balanza Comercial"]) == 300.0


def test_parse_indec_dataframe_prefers_period_month_over_fecha() -> None:
    raw = pd.DataFrame(
        {
            "Periodo": [2025, 2025],
            "Mes": ["enero*", "febrero*"],
            "Fecha": ["2022-01-01", "2022-02-01"],
            "Exportaciones": [1500, 1600],
            "Importaciones": [1200, 1300],
        }
    )
    parsed = macro._parse_indec_dataframe(raw)
    assert not parsed.empty
    assert list(parsed.index.strftime("%Y-%m-%d")) == ["2025-01-01", "2025-02-01"]


def test_parse_indec_dataframe_picks_most_recent_between_column_combinations() -> None:
    raw = pd.DataFrame(
        {
            "Periodo viejo": [2022, 2022, 2022],
            "Mes viejo": ["enero", "febrero", "marzo"],
            "Exportaciones viejo": [100, 101, 102],
            "Importaciones viejo": [90, 91, 92],
            "Periodo": [2025, 2025, 2025],
            "Mes": ["enero", "febrero", "marzo"],
            "Exportaciones": [200, 201, 202],
            "Importaciones": [180, 181, 182],
        }
    )
    parsed = macro._parse_indec_dataframe(raw)
    assert not parsed.empty
    assert parsed.index.max() == pd.Timestamp("2025-03-01")
    assert float(parsed.iloc[-1]["Exportaciones"]) == 202.0


def test_parse_indec_dataframe_detects_year_month_columns_by_content() -> None:
    raw = pd.DataFrame(
        {
            "Unnamed: 0": [2025, None, None],
            "Unnamed: 1": ["enero*", "febrero*", "marzo*"],
            "Exportaciones Total mensual": ["1.100,0", "1.200,0", "1.300,0"],
            "Importaciones Total mensual": ["1.000,0", "1.050,0", "1.100,0"],
        }
    )
    parsed = macro._parse_indec_dataframe(raw)
    assert not parsed.empty
    assert list(parsed.index.strftime("%Y-%m-%d")) == ["2025-01-01", "2025-02-01", "2025-03-01"]
    assert float(parsed.iloc[-1]["Balanza Comercial"]) == 200.0


def test_parse_indec_dataframe_returns_empty_if_key_columns_missing() -> None:
    raw = pd.DataFrame({"Foo": [1, 2], "Bar": [3, 4]})
    parsed = macro._parse_indec_dataframe(raw)
    assert parsed.empty


def test_get_indec_trade_balance_selects_most_recent_candidate(monkeypatch) -> None:
    class DummyResponse:
        content = b"xls-bytes"

        @staticmethod
        def raise_for_status() -> None:
            return None

    monkeypatch.setattr(macro.requests, "get", lambda url, timeout=30: DummyResponse())

    def fake_read_excel(*args, **kwargs):
        return pd.DataFrame({"dummy": [1]})

    monkeypatch.setattr(macro.pd, "read_excel", fake_read_excel)

    candidates = [
        pd.DataFrame(
            {
                "Exportaciones": [100.0],
                "Importaciones": [90.0],
                "Balanza Comercial": [10.0],
            },
            index=pd.to_datetime(["2022-12-01"]),
        ),
        pd.DataFrame(
            {
                "Exportaciones": [120.0, 130.0],
                "Importaciones": [100.0, 110.0],
                "Balanza Comercial": [20.0, 20.0],
            },
            index=pd.to_datetime(["2025-01-01", "2025-02-01"]),
        ),
    ]

    def fake_parse(_raw):
        if candidates:
            return candidates.pop(0)
        return pd.DataFrame()

    monkeypatch.setattr(macro, "_parse_indec_dataframe", fake_parse)
    result = macro.get_indec_trade_balance()
    assert result.dataframe.index.max() == pd.Timestamp("2025-02-01")
