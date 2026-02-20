from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from portfolio_analyzer.data.macro import bcra_variables, get_bcra_series


@st.cache_data(ttl=1800)
def load_bcra_series(
    variable_id: int,
    start_date_iso: str,
    end_date_iso: str,
    adjustment: str,
):
    return get_bcra_series(
        variable_id=variable_id,
        start_date=start_date_iso,
        end_date=end_date_iso,
        adjustment=adjustment,
    )


st.title("BCRA Series Analysis")
st.caption("Consulta de series monetarias con visualizacion directa en Streamlit.")
st.info(
    "Selecciona una serie para seguimiento diario o compara dos series para ver co-movimientos.",
    icon=":material/info:",
)
if st.button("Refrescar cache BCRA", icon=":material/refresh:"):
    st.cache_data.clear()
    st.rerun()

variables = bcra_variables()
labels = {f"{key} - {value}": key for key, value in variables.items()}
all_options = sorted(labels.keys())

with st.form("bcra_form"):
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Desde", value=date(2020, 1, 1))
    with col2:
        end_date = st.date_input("Hasta", value=date.today())

    filter_text = st.text_input(
        "Filtrar series por texto",
        value="",
        placeholder="Ej: reservas, base monetaria, dolar, tasa",
    ).strip()
    filtered_options = [
        option for option in all_options if filter_text.lower() in option.lower()
    ] or all_options

    primary_label = st.selectbox("Serie principal", options=filtered_options)
    compare_enabled = st.checkbox("Comparar con segunda serie", value=False)
    preview_rows = st.slider("Filas a mostrar", min_value=10, max_value=180, value=30, step=10)

    secondary_label = None
    if compare_enabled:
        secondary_label = st.selectbox(
            "Serie secundaria",
            options=[k for k in filtered_options if k != primary_label],
        )

    adjustment = "none"
    if not compare_enabled:
        adjustment = st.selectbox(
            "Ajuste",
            options=["none", "usd", "cer"],
            format_func=lambda x: {"none": "Sin ajuste", "usd": "USD constante", "cer": "CER"}[x],
        )

    submit = st.form_submit_button("Consultar BCRA", type="primary")

if submit:
    if start_date > end_date:
        st.error("La fecha 'Desde' no puede ser mayor que 'Hasta'.")
        st.stop()

    start_iso = start_date.strftime("%Y-%m-%d")
    end_iso = end_date.strftime("%Y-%m-%d")

    with st.spinner("Descargando datos del BCRA..."):
        try:
            primary = load_bcra_series(
                variable_id=labels[primary_label],
                start_date_iso=start_iso,
                end_date_iso=end_iso,
                adjustment=adjustment,
            )
        except Exception as exc:
            st.error(f"Error obteniendo serie principal: {exc}")
            st.stop()

        if compare_enabled and secondary_label:
            try:
                secondary = load_bcra_series(
                    variable_id=labels[secondary_label],
                    start_date_iso=start_iso,
                    end_date_iso=end_iso,
                    adjustment="none",
                )
            except Exception as exc:
                st.error(f"Error obteniendo serie secundaria: {exc}")
                st.stop()

            merged = primary.dataframe.rename(columns={"valor": primary_label}).join(
                secondary.dataframe.rename(columns={"valor": secondary_label}),
                how="inner",
            )
            if merged.empty:
                st.warning("No hay solapamiento de fechas entre ambas series.")
                st.stop()
            st.metric("Observaciones comparables", int(len(merged)))
            st.line_chart(merged)
            st.dataframe(merged.tail(preview_rows), use_container_width=True)
            st.download_button(
                "Descargar CSV (comparacion)",
                data=merged.to_csv(index=True).encode("utf-8"),
                file_name="bcra_series_comparison.csv",
                mime="text/csv",
            )
        else:
            st.line_chart(primary.dataframe)
            latest = float(primary.dataframe["valor"].iloc[-1])
            previous = (
                float(primary.dataframe["valor"].iloc[-2]) if len(primary.dataframe) > 1 else latest
            )
            delta_pct = ((latest / previous) - 1) * 100 if previous else 0.0
            col1, col2 = st.columns(2)
            col1.metric("Ultimo valor", f"{latest:,.4f}")
            col2.metric("Cambio ultimo dato", f"{delta_pct:.2f}%")
            st.dataframe(primary.dataframe.tail(preview_rows), use_container_width=True)
            st.download_button(
                "Descargar CSV",
                data=primary.dataframe.to_csv(index=True).encode("utf-8"),
                file_name="bcra_series.csv",
                mime="text/csv",
            )
