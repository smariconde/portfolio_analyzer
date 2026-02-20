from __future__ import annotations

import pandas as pd

from portfolio_analyzer.data import macro


def test_month_to_number_known_values() -> None:
    assert macro._month_to_number("enero") == 1
    assert macro._month_to_number("sep") == 9
    assert macro._month_to_number("DICIEMBRE") == 12


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


def test_parse_indec_dataframe_returns_empty_if_key_columns_missing() -> None:
    raw = pd.DataFrame({"Foo": [1, 2], "Bar": [3, 4]})
    parsed = macro._parse_indec_dataframe(raw)
    assert parsed.empty
