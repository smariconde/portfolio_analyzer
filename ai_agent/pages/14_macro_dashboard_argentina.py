from __future__ import annotations

import sys
from datetime import date
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

from portfolio_analyzer.data.argentina_dashboard import (
    SeriesResult,
    get_ccl_inflation_wages,
    get_emae_series,
    get_fiscal_result_pct_gdp,
    get_fx_vs_itcrm,
    get_palermo_m2,
    get_salary_index,
    get_soy_real_price,
    get_trade_balance_annual,
)


st.title("Macro Dashboard Argentina")
st.caption("Vista unica con 8 graficos macro para monitoreo y comparacion.")

if "macro_refresh_token" not in st.session_state:
    st.session_state.macro_refresh_token = 0

with st.sidebar:
    st.subheader("Controles")
    since = st.date_input("Desde", value=date(2001, 1, 1))
    until = st.date_input("Hasta", value=date.today())
    base_date = st.date_input("Base de indexacion", value=date(2026, 2, 1))
    milei_start = st.date_input("Inicio mandato", value=date(2023, 12, 1))
    layout_mode = st.radio(
        "Layout",
        options=["2 columnas", "1 columna (amplio)"],
        index=0,
        horizontal=False,
    )
    chart_height = st.slider("Alto de graficos", min_value=260, max_value=560, value=360, step=20)
    if st.button("Refrescar cache", type="primary"):
        st.session_state.macro_refresh_token += 1
        st.cache_data.clear()
        st.rerun()


@st.cache_data(ttl=3600)
def load_all(refresh_token: int, base_date_iso: str, milei_start_iso: str):
    _ = refresh_token

    def safe_call(fetcher, *args, **kwargs) -> SeriesResult:
        try:
            return fetcher(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return SeriesResult(dataframe=pd.DataFrame(), source=f"Error: {exc}")

    return {
        "emae": safe_call(get_emae_series),
        "mandato": safe_call(get_ccl_inflation_wages, base_date=milei_start_iso),
        "fx_itcrm": safe_call(get_fx_vs_itcrm, base_date=base_date_iso),
        "fiscal": safe_call(get_fiscal_result_pct_gdp),
        "salary": safe_call(get_salary_index, base_date=base_date_iso),
        "trade": safe_call(get_trade_balance_annual),
        "soy": safe_call(get_soy_real_price, base_date=base_date_iso),
        "m2": safe_call(get_palermo_m2),
    }


def _slice_date(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return df


def _show_unavailable(msg: str):
    st.warning(msg, icon=":material/warning:")


def _reset_with_date(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.reset_index()
    return out, out.columns[0]


with st.spinner("Cargando series macro..."):
    if since > until:
        st.error("La fecha 'Desde' no puede ser mayor que 'Hasta'.")
        st.stop()

    payload = load_all(
        st.session_state.macro_refresh_token,
        base_date.strftime("%Y-%m-%d"),
        milei_start.strftime("%Y-%m-%d"),
    )

single_column = layout_mode == "1 columna (amplio)"
if single_column:
    row1 = st.columns(1)
    row2 = st.columns(1)
    row3 = st.columns(1)
    row4 = st.columns(1)
    row5 = st.columns(1)
    row6 = st.columns(1)
    row7 = st.columns(1)
    row8 = st.columns(1)
else:
    row1 = st.columns(2)
    row2 = st.columns(2)
    row3 = st.columns(2)
    row4 = st.columns(2)

with row1[0]:
    st.subheader("1) EMAE mensual")
    emae = payload["emae"].dataframe.copy()
    emae = _slice_date(emae, since, until)
    if emae.empty or "EMAE" not in emae:
        _show_unavailable("No se pudo obtener EMAE automaticamente.")
    else:
        emae_reset, date_col = _reset_with_date(emae)
        chart = (
            alt.Chart(emae_reset)
            .mark_line(strokeWidth=2.5, color="#2563EB")
            .encode(x=alt.X(f"{date_col}:T", title="Fecha"), y=alt.Y("EMAE:Q", title="Indice"))
            .properties(height=chart_height)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")
        st.caption(f"Fuente: {payload['emae'].source or 'n/d'}")

with (row2[0] if single_column else row1[1]):
    st.subheader("2) CCL vs Inflacion vs Salarios")
    mandato = payload["mandato"].dataframe.copy()
    mandate_start = max(since, milei_start)
    mandato = _slice_date(mandato, mandate_start, until)
    if mandato.empty:
        _show_unavailable("No se pudo construir el panel de mandato (CCL/Inflacion/Salarios).")
    else:
        keep_cols = [c for c in ["CCL", "Inflacion", "Salarios"] if c in mandato.columns]
        mandato_reset, date_col = _reset_with_date(mandato[keep_cols])
        chart_df = mandato_reset.melt(id_vars=[date_col], var_name="Serie", value_name="Indice")
        chart = (
            alt.Chart(chart_df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X(f"{date_col}:T", title="Fecha"),
                y=alt.Y("Indice:Q", title="Base=1"),
                color=alt.Color("Serie:N", scale=alt.Scale(range=["#22C55E", "#EF4444", "#EAB308"])),
            )
            .properties(height=chart_height)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")
        st.caption(f"Fuente: {payload['mandato'].source or 'n/d'}")

with (row3[0] if single_column else row2[0]):
    st.subheader("3) Oficial y CCL ajustado por ITCRM")
    fx = payload["fx_itcrm"].dataframe.copy()
    fx = _slice_date(fx, since, until)
    cols = [
        c
        for c in ["Oficial_Ajustado_ITCRM", "CCL_Ajustado_ITCRM"]
        if c in fx.columns
    ]
    if fx.empty or not cols:
        _show_unavailable("No se pudo construir Oficial/CCL ajustado por ITCRM.")
    else:
        fx_plot = fx[cols].dropna(how="all")
        if fx_plot.empty:
            _show_unavailable("No hay datos para el rango seleccionado en Oficial/CCL ajustado por ITCRM.")
        else:
            st.line_chart(fx_plot, height=chart_height, width="stretch")
        st.caption(f"Fuente: {payload['fx_itcrm'].source or 'n/d'}")

with (row4[0] if single_column else row2[1]):
    st.subheader("4) Resultado fiscal (% PBI)")
    fiscal = payload["fiscal"].dataframe.copy()
    fiscal = fiscal[(fiscal.index >= since.year) & (fiscal.index <= until.year)]
    cols = [c for c in ["Resultado_Primario_Pct_PBI", "Resultado_Financiero_Pct_PBI"] if c in fiscal.columns]
    if fiscal.empty or not cols:
        _show_unavailable(
            "No hay serie fiscal disponible. Si queres, agrego `data/fiscal_resultado_pct_pbi.csv`."
        )
    else:
        f = fiscal.reset_index().melt(id_vars=["Anio"], var_name="Serie", value_name="PctPBI")
        bars = (
            alt.Chart(f[f["Serie"] == "Resultado_Primario_Pct_PBI"])
            .mark_bar(color="#22C55E")
            .encode(x=alt.X("Anio:O", title="Anio"), y=alt.Y("PctPBI:Q", title="% PBI"))
        )
        line = (
            alt.Chart(f[f["Serie"] == "Resultado_Financiero_Pct_PBI"])
            .mark_line(color="#EF4444", point=True)
            .encode(x=alt.X("Anio:O"), y=alt.Y("PctPBI:Q"))
        )
        st.altair_chart((bars + line).properties(height=chart_height), width="stretch")
        st.caption(f"Fuente: {payload['fiscal'].source or 'n/d'}")

with (row5[0] if single_column else row3[0]):
    st.subheader("5) Indice salarios registrados")
    sal = payload["salary"].dataframe.copy()
    sal = _slice_date(sal, since, until)
    if sal.empty or "Indice_Salarios_Real_Base" not in sal:
        _show_unavailable("No se pudo obtener la serie de salarios.")
    else:
        sal_plot = sal[["Indice_Salarios_Real_Base"]].copy().dropna()
        if sal_plot.empty:
            _show_unavailable("No hay datos para el rango seleccionado en salarios reales.")
        else:
            st.line_chart(sal_plot, height=chart_height, width="stretch")
        st.caption(f"Fuente: {payload['salary'].source or 'n/d'}")

with (row6[0] if single_column else row3[1]):
    st.subheader("6) Balanza comercial desde 1990")
    trade = payload["trade"].dataframe.copy()
    trade = trade[(trade.index >= since.year) & (trade.index <= until.year)]
    if trade.empty:
        _show_unavailable("No se pudo obtener INDEC balanza comercial.")
    else:
        t = trade.reset_index().rename(columns={"Anio": "year"})
        bars = (
            alt.Chart(t)
            .mark_bar(color="#0EA5E9")
            .encode(
                x=alt.X("year:O", title="Anio"),
                y=alt.Y("Balanza Comercial:Q", title="Saldo (Millones USD)"),
            )
        )
        lines = (
            alt.Chart(t)
            .transform_fold(["Exportaciones", "Importaciones"], as_=["Serie", "Valor"])
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("year:O"),
                y=alt.Y("Valor:Q", title="Expo/Impo (Millones USD)"),
                color=alt.Color("Serie:N", scale=alt.Scale(range=["#16A34A", "#EF4444"])),
            )
        )
        trade_chart = (
            alt.layer(bars, lines).resolve_scale(y="independent").properties(height=chart_height).interactive()
        )
        st.altair_chart(trade_chart, width="stretch")
        st.caption(f"Fuente: {payload['trade'].source or 'n/d'}")

with (row7[0] if single_column else row4[0]):
    st.subheader("7) Soja ajustada por inflacion")
    soy = payload["soy"].dataframe.copy()
    soy = _slice_date(soy, since, until)
    cols = [c for c in ["Soja_Real"] if c in soy.columns]
    if soy.empty or "Soja_Real" not in soy.columns:
        _show_unavailable("No se pudo obtener soja ajustada.")
    else:
        soy_plot = soy[["Soja_Real"]].dropna()
        st.line_chart(soy_plot, height=chart_height, width="stretch")
        st.caption(f"Fuente: {payload['soy'].source or 'n/d'}")

with (row8[0] if single_column else row4[1]):
    st.subheader("8) Evolucion M2 Palermo")
    m2 = payload["m2"].dataframe.copy()
    m2 = _slice_date(m2, since, until)
    if m2.empty or "M2_Palermo" not in m2:
        _show_unavailable("No se pudo obtener M2 Palermo automaticamente.")
    else:
        m2_reset, date_col = _reset_with_date(m2)
        chart = (
            alt.Chart(m2_reset)
            .mark_line(strokeWidth=2.5, color="#2563EB")
            .encode(x=alt.X(f"{date_col}:T", title="Fecha"), y=alt.Y("M2_Palermo:Q", title="USD/m2"))
            .properties(height=chart_height)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")
        st.caption(f"Fuente: {payload['m2'].source or 'n/d'}")

st.divider()
st.caption(
    "Nota: algunos endpoints externos pueden cambiar. Cuando una fuente no responde, el panel avisa y sigue cargando el resto."
)

with st.expander("Diagnostico de fuentes"):
    diag_rows = []
    for key, result in payload.items():
        source = result.source or "n/d"
        status = "OK"
        if source.lower().startswith("error:") or result.dataframe.empty:
            status = "Warning"
        diag_rows.append(
            {
                "panel": key,
                "status": status,
                "filas": int(len(result.dataframe)),
                "fuente_o_error": source,
            }
        )
    st.dataframe(pd.DataFrame(diag_rows), width="stretch")
