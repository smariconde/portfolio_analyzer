from __future__ import annotations

import sys
from pathlib import Path

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

if st.button("Actualizar datos INDEC", type="primary"):
    st.cache_data.clear()


@st.cache_data(ttl=3600)
def load_data():
    return get_indec_trade_balance()


with st.spinner("Descargando y procesando archivo del INDEC..."):
    result = None
    try:
        result = load_data()
    except Exception as exc:
        st.error(f"Error en datos de INDEC: {exc}")

if result is not None:
    st.caption(f"Fuente: {result.source_url}")
    months_to_show = st.slider("Meses a mostrar", min_value=12, max_value=120, value=36, step=12)
    visible = result.dataframe.tail(months_to_show).copy()
    latest = result.dataframe.iloc[-1]
    yoy_export = None
    yoy_import = None
    if len(result.dataframe) >= 13:
        prev = result.dataframe.iloc[-13]
        yoy_export = ((latest["Exportaciones"] / prev["Exportaciones"]) - 1) * 100
        yoy_import = ((latest["Importaciones"] / prev["Importaciones"]) - 1) * 100

    col1, col2, col3 = st.columns(3)
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

    st.line_chart(visible[["Exportaciones", "Importaciones"]])
    st.line_chart(visible[["Balanza Comercial"]])
    st.dataframe(visible, use_container_width=True)
    st.download_button(
        "Descargar CSV INDEC",
        data=result.dataframe.to_csv(index=True).encode("utf-8"),
        file_name="indec_trade_balance.csv",
        mime="text/csv",
    )
