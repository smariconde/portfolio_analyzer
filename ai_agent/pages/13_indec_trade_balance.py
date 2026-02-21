from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from portfolio_analyzer.data.macro import get_indec_trade_balance

st.title("INDEC Trade Balance")
st.caption("Serie historica de exportaciones, importaciones y balanza comercial.")
st.info(
    "Panel rapido para monitorear comercio exterior argentino y descargar la serie consolidada.",
    icon=":material/info:",
)

if "indec_refresh_token" not in st.session_state:
    st.session_state.indec_refresh_token = 0

if st.button("Actualizar datos INDEC", type="primary"):
    st.session_state.indec_refresh_token += 1
    st.cache_data.clear()
    st.rerun()


@st.cache_data(ttl=3600)
def load_data(refresh_token: int):
    _ = refresh_token
    return get_indec_trade_balance()


with st.spinner("Descargando y procesando archivo del INDEC..."):
    result = None
    try:
        result = load_data(st.session_state.indec_refresh_token)
    except Exception as exc:
        st.error(f"Error en datos de INDEC: {exc}")

if result is not None:
    st.caption(f"Fuente: {result.source_url}")
    latest_date = result.dataframe.index.max()
    days_since_latest = (pd.Timestamp.now().normalize() - latest_date.normalize()).days
    st.caption(f"Ultimo dato disponible: {latest_date:%Y-%m}")
    if days_since_latest > 70:
        st.warning(
            "La serie puede estar desactualizada respecto del calendario mensual esperado del INDEC.",
            icon=":material/warning:",
        )

    months_to_show = st.slider(
        "Meses a mostrar",
        min_value=6,
        max_value=min(240, len(result.dataframe)),
        value=min(48, len(result.dataframe)),
        step=1,
    )
    quick_range = st.radio(
        "Rango rapido",
        options=["24m", "48m", "120m", "Todo"],
        horizontal=True,
        index=1,
    )
    if quick_range == "Todo":
        months_to_show = len(result.dataframe)
    else:
        months_to_show = min(int(quick_range.replace("m", "")), len(result.dataframe))
    visible = result.dataframe.tail(months_to_show).copy()
    latest = result.dataframe.iloc[-1]
    yoy_export = None
    yoy_import = None
    if len(result.dataframe) >= 13:
        prev = result.dataframe.iloc[-13]
        yoy_export = ((latest["Exportaciones"] / prev["Exportaciones"]) - 1) * 100
        yoy_import = ((latest["Importaciones"] / prev["Importaciones"]) - 1) * 100

    rolling_12 = result.dataframe["Balanza Comercial"].rolling(12).sum()
    rolling_3 = result.dataframe["Balanza Comercial"].rolling(3).mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Exportaciones (ult)",
        f"{latest['Exportaciones']:,.0f}",
        f"{yoy_export:.2f}% YoY" if yoy_export is not None else None,
    )
    col2.metric(
        "Importaciones (ult)",
        f"{latest['Importaciones']:,.0f}",
        f"{yoy_import:.2f}% YoY" if yoy_import is not None else None,
    )
    col3.metric("Balanza (ult)", f"{latest['Balanza Comercial']:,.0f}")
    col4.metric("Balanza 12m acum", f"{rolling_12.iloc[-1]:,.0f}")

    chart_df = visible.reset_index().rename(columns={"Fecha": "fecha"})
    chart_long = chart_df.melt(
        id_vars=["fecha"],
        value_vars=["Exportaciones", "Importaciones"],
        var_name="serie",
        value_name="valor",
    )

    st.subheader("Exportaciones vs Importaciones (mensual)")
    trade_chart = (
        alt.Chart(chart_long)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("valor:Q", title="Millones de USD"),
            color=alt.Color(
                "serie:N",
                title="Serie",
                scale=alt.Scale(
                    domain=["Exportaciones", "Importaciones"],
                    range=["#0A84FF", "#FF6B00"],
                ),
            ),
            tooltip=[
                alt.Tooltip("fecha:T", title="Mes"),
                alt.Tooltip("serie:N", title="Serie"),
                alt.Tooltip("valor:Q", title="Valor", format=",.0f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(trade_chart, use_container_width=True)

    st.subheader("Balanza Comercial (mensual)")
    chart_df["balance_color"] = chart_df["Balanza Comercial"].apply(
        lambda value: "Superavit" if value >= 0 else "Deficit"
    )
    balance_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("Balanza Comercial:Q", title="Millones de USD"),
            color=alt.Color(
                "balance_color:N",
                title="Resultado",
                scale=alt.Scale(domain=["Superavit", "Deficit"], range=["#16A34A", "#DC2626"]),
            ),
            tooltip=[
                alt.Tooltip("fecha:T", title="Mes"),
                alt.Tooltip("Balanza Comercial:Q", title="Balanza", format=",.0f"),
            ],
        )
        .properties(height=320)
    )
    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#6B7280", strokeDash=[6, 4]).encode(y="y:Q")
    st.altair_chart(balance_chart + zero_line, use_container_width=True)

    st.subheader("Tendencia Balanza (promedio 3 meses)")
    rolling_3_visible = rolling_3.tail(months_to_show).reset_index()
    rolling_3_visible.columns = ["fecha", "valor"]
    trend_chart = (
        alt.Chart(rolling_3_visible)
        .mark_line(strokeWidth=3, color="#7C3AED")
        .encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("valor:Q", title="Millones de USD"),
            tooltip=[
                alt.Tooltip("fecha:T", title="Mes"),
                alt.Tooltip("valor:Q", title="Promedio 3m", format=",.0f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(trend_chart, use_container_width=True)
    st.dataframe(visible, use_container_width=True)
    st.download_button(
        "Descargar CSV INDEC",
        data=result.dataframe.to_csv(index=True).encode("utf-8"),
        file_name="indec_trade_balance.csv",
        mime="text/csv",
    )
