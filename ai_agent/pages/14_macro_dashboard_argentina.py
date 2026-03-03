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
    get_fiscal_monthly_result,
    get_fiscal_nowcast_pct_gdp,
    get_fiscal_result_pct_gdp,
    get_fx_vs_itcrm,
    get_inflation_and_expectations,
    get_macro_regime_markers,
    get_monetary_liquidity,
    get_real_rates_panel,
    get_salary_index,
    get_soy_real_price,
    get_trade_balance_annual,
)


st.title("Macro Dashboard Argentina")
st.caption("Panel macro-financiero de Argentina para decision de asset allocation local.")

if "macro_refresh_token" not in st.session_state:
    st.session_state.macro_refresh_token = 0

with st.sidebar:
    st.subheader("Controles")
    since = st.date_input("Desde", value=date(2001, 1, 1))
    until = st.date_input("Hasta", value=date.today())
    base_date = st.date_input("Base de indexacion", value=date.today().replace(day=1))
    chart_height = st.slider("Alto de graficos", min_value=280, max_value=620, value=380, step=20)
    fiscal_year_window = st.slider("Ventana fiscal anual (anios)", min_value=8, max_value=40, value=15, step=1)
    if st.button("Refrescar cache", type="primary"):
        st.session_state.macro_refresh_token += 1
        st.cache_data.clear()
        st.rerun()


@st.cache_data(ttl=3600)
def load_all(refresh_token: int, base_date_iso: str):
    _ = refresh_token

    def safe_call(fetcher, *args, **kwargs) -> SeriesResult:
        try:
            return fetcher(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return SeriesResult(dataframe=pd.DataFrame(), source=f"Error: {exc}", status="error")

    return {
        "emae": safe_call(get_emae_series),
        "infl_exp": safe_call(get_inflation_and_expectations),
        "mandato": safe_call(get_ccl_inflation_wages, base_date="2023-12-01"),
        "fx_itcrm": safe_call(get_fx_vs_itcrm, base_date=base_date_iso),
        "fiscal": safe_call(get_fiscal_result_pct_gdp),
        "fiscal_monthly": safe_call(get_fiscal_monthly_result),
        "fiscal_nowcast": safe_call(get_fiscal_nowcast_pct_gdp),
        "salary": safe_call(get_salary_index, base_date=base_date_iso),
        "trade": safe_call(get_trade_balance_annual),
        "liquidity": safe_call(get_monetary_liquidity),
        "rates": safe_call(get_real_rates_panel),
        "soy": safe_call(get_soy_real_price, base_date=base_date_iso),
    }


def _slice_date(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        return df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return df


def _slice_year(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df.index >= start_year) & (df.index <= end_year)]


def _reset_with_date(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.reset_index()
    return out, out.columns[0]


def _show_unavailable(msg: str):
    st.warning(msg, icon=":material/warning:")


def _add_markers(chart: alt.Chart, date_col: str, markers: pd.DataFrame, max_labels: int = 4) -> alt.LayerChart:
    if markers.empty:
        return chart
    rules = (
        alt.Chart(markers)
        .mark_rule(color="#9CA3AF", strokeDash=[4, 4])
        .encode(x=alt.X("fecha:T"))
    )
    labels = (
        alt.Chart(markers.head(max_labels))
        .mark_text(align="left", baseline="top", dx=4, dy=4, color="#6B7280", fontSize=10)
        .encode(x=alt.X("fecha:T"), y=alt.value(8), text=alt.Text("evento:N"))
    )
    return alt.layer(chart, rules, labels)


def _explain(what: str, why: str, how: str):
    st.markdown(f"**Que mide:** {what}")
    st.markdown(f"**Por que importa:** {why}")
    st.markdown(f"**Como leerlo para decision:** {how}")


with st.spinner("Cargando series macro..."):
    if since > until:
        st.error("La fecha 'Desde' no puede ser mayor que 'Hasta'.")
        st.stop()

    payload = load_all(
        st.session_state.macro_refresh_token,
        base_date.strftime("%Y-%m-%d"),
    )

markers = get_macro_regime_markers()
markers = markers[(markers["fecha"] >= pd.Timestamp(since)) & (markers["fecha"] <= pd.Timestamp(until))]

st.subheader("1) Actividad real (EMAE mensual)")
emae = _slice_date(payload["emae"].dataframe.copy(), since, until)
if emae.empty or "EMAE" not in emae:
    _show_unavailable("No se pudo obtener EMAE automaticamente.")
else:
    emae_reset, date_col = _reset_with_date(emae)
    base_chart = (
        alt.Chart(emae_reset)
        .mark_line(strokeWidth=2.6, color="#2563EB")
        .encode(x=alt.X(f"{date_col}:T", title="Fecha"), y=alt.Y("EMAE:Q", title="Indice"))
        .properties(height=chart_height)
        .interactive()
    )
    st.altair_chart(_add_markers(base_chart, date_col, markers), width="stretch")
    _explain(
        "Nivel de actividad economica desestacionalizada.",
        "Anticipa recaudacion, utilidades corporativas y riesgo de credito.",
        "Pendiente positiva sostenida favorece riesgo local; desaceleracion fuerte sugiere posicion mas defensiva.",
    )
    st.caption(f"Fuente: {payload['emae'].source or 'n/d'}")

st.subheader("2) Inflacion observada vs expectativa REM")
infl = _slice_date(payload["infl_exp"].dataframe.copy(), since, until)
if infl.empty:
    _show_unavailable("No se pudo obtener inflacion/REM desde BCRA.")
else:
    cols = [c for c in ["Inflacion_Mensual_Pct", "Inflacion_Interanual_Pct", "REM_Inflacion_12m_Pct"] if c in infl]
    if not cols:
        _show_unavailable("No hay columnas de inflacion disponibles.")
    else:
        infl_reset, date_col = _reset_with_date(infl[cols])
        infl_m = infl_reset.melt(id_vars=[date_col], var_name="Serie", value_name="Pct")
        chart = (
            alt.Chart(infl_m)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X(f"{date_col}:T", title="Fecha"),
                y=alt.Y("Pct:Q", title="%"),
                color=alt.Color("Serie:N", scale=alt.Scale(range=["#EF4444", "#F59E0B", "#10B981"])),
            )
            .properties(height=chart_height)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")
        _explain(
            "Inflacion mensual, interanual y expectativa de 12 meses (REM).",
            "Define tasa real, valuacion de bonos CER/fija y riesgo de licuacion nominal.",
            "Si REM baja y la nominalidad baja mas rapido, mejora escenario para duration en pesos; si REM se desancla, sube cobertura.",
        )
        st.caption(f"Fuente: {payload['infl_exp'].source or 'n/d'}")

st.subheader("3) CCL vs Inflacion vs Salarios (base dic-2023)")
mandato = _slice_date(payload["mandato"].dataframe.copy(), max(since, date(2023, 12, 1)), until)
if mandato.empty:
    _show_unavailable("No se pudo construir el panel CCL/Inflacion/Salarios.")
else:
    keep_cols = [c for c in ["CCL", "Inflacion", "Salarios"] if c in mandato.columns]
    if keep_cols:
        m_reset, date_col = _reset_with_date(mandato[keep_cols])
        chart_df = m_reset.melt(id_vars=[date_col], var_name="Serie", value_name="Indice")
        chart = (
            alt.Chart(chart_df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X(f"{date_col}:T", title="Fecha"),
                y=alt.Y("Indice:Q", title="Base = 1"),
                color=alt.Color("Serie:N", scale=alt.Scale(range=["#22C55E", "#EF4444", "#EAB308"])),
            )
            .properties(height=chart_height)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")
        _explain(
            "Evolucion relativa del dolar financiero, precios y salarios registrados.",
            "Permite evaluar si hay atraso/correccion cambiaria y recomposicion o deterioro de ingreso real.",
            "CCL arriba de salarios e inflacion sugiere dolarizacion preventiva; salarios reales recuperando favorecen consumo domestico.",
        )
        st.caption(f"Fuente: {payload['mandato'].source or 'n/d'}")

st.subheader("4) Tipo de cambio oficial y CCL ajustados por ITCRM")
fx = _slice_date(payload["fx_itcrm"].dataframe.copy(), since, until)
cols = [c for c in ["Oficial_Ajustado_ITCRM_Base", "CCL_Ajustado_ITCRM_Base"] if c in fx.columns]
if fx.empty or not cols:
    _show_unavailable("No se pudo construir Oficial/CCL ajustado por ITCRM.")
else:
    st.line_chart(fx[cols].dropna(how="all"), height=chart_height, width="stretch")
    latest = fx[cols].dropna(how="all")
    if not latest.empty and len(cols) == 2:
        spread = float(latest[cols[1]].iloc[-1] / latest[cols[0]].iloc[-1] - 1) * 100
        st.metric("Brecha real CCL vs oficial ajustado", f"{spread:.1f}%")
    _explain(
        "Paridad cambiaria en terminos reales para oficial y CCL.",
        "Ayuda a detectar atraso o sobre-reaccion del FX frente a competitividad externa.",
        "Brecha real en compresion reduce premio de cobertura; brecha ampliando alerta sobre riesgo de salto nominal.",
    )
    st.caption(f"Fuente: {payload['fx_itcrm'].source or 'n/d'}")

st.subheader("5) Resultado fiscal: historico % PBI + seguimiento mensual")
fiscal = _slice_year(payload["fiscal"].dataframe.copy(), since.year, until.year)
monthly_fiscal = _slice_date(payload["fiscal_monthly"].dataframe.copy(), since, until)

f_cols = [c for c in ["Resultado_Primario_Pct_PBI", "Resultado_Financiero_Pct_PBI"] if c in fiscal.columns]
if not fiscal.empty and f_cols:
    latest_year = int(fiscal.index.max())
    fiscal_recent = fiscal[fiscal.index >= max(since.year, latest_year - fiscal_year_window + 1)].copy()
    f = fiscal_recent.reset_index().melt(id_vars=["Anio"], var_name="Serie", value_name="PctPBI")
    bars = (
        alt.Chart(f[f["Serie"] == "Resultado_Primario_Pct_PBI"])
        .mark_bar(color="#22C55E", size=26)
        .encode(
            x=alt.X("Anio:O", title="Anio"),
            y=alt.Y("PctPBI:Q", title="% PBI"),
            tooltip=[alt.Tooltip("Anio:O", title="Anio"), alt.Tooltip("PctPBI:Q", title="% PBI", format=".2f")],
        )
    )
    line = (
        alt.Chart(f[f["Serie"] == "Resultado_Financiero_Pct_PBI"])
        .mark_line(color="#EF4444", point=True)
        .encode(
            x=alt.X("Anio:O"),
            y=alt.Y("PctPBI:Q"),
            tooltip=[alt.Tooltip("Anio:O", title="Anio"), alt.Tooltip("PctPBI:Q", title="% PBI", format=".2f")],
        )
    )
    st.altair_chart((bars + line).properties(height=chart_height), width="stretch")
    st.caption(
        f"Historico %PBI hasta {latest_year} (ultimo disponible en PIB oficial usado por la serie)."
    )
else:
    _show_unavailable("No hay serie anual fiscal %PBI disponible.")

if monthly_fiscal.empty:
    _show_unavailable("No se pudo obtener seguimiento fiscal mensual actualizado.")
else:
    m = monthly_fiscal.copy()
    m_recent = m[m.index >= (m.index.max() - pd.DateOffset(months=36))]
    m_cols = [c for c in ["Resultado_Primario_12m_Mill_ARS", "Resultado_Financiero_12m_Mill_ARS"] if c in m_recent]
    if m_cols:
        st.line_chart(m_recent[m_cols].dropna(how="all"), height=300, width="stretch")
    last_m = m.index.max().date()
    if "Resultado_Primario_Mill_ARS" in m.columns and "Resultado_Financiero_Mill_ARS" in m.columns:
        l = m[["Resultado_Primario_Mill_ARS", "Resultado_Financiero_Mill_ARS"]].dropna().iloc[-1]
        st.metric("Ultimo primario mensual (MM ARS)", f"{float(l['Resultado_Primario_Mill_ARS']):,.0f}")
        st.metric("Ultimo financiero mensual (MM ARS)", f"{float(l['Resultado_Financiero_Mill_ARS']):,.0f}")
    st.caption(f"Seguimiento mensual actualizado hasta {last_m}.")

nowcast = _slice_date(payload["fiscal_nowcast"].dataframe.copy(), since, until)
if nowcast.empty:
    _show_unavailable("No se pudo construir nowcast fiscal %PIB actualizado.")
else:
    n_cols = [
        c
        for c in ["Resultado_Primario_12m_Pct_PIB_Nowcast", "Resultado_Financiero_12m_Pct_PIB_Nowcast"]
        if c in nowcast.columns
    ]
    if n_cols:
        st.line_chart(nowcast[n_cols].dropna(how="all"), height=300, width="stretch")
        n_last = nowcast[n_cols].dropna(how="all")
        if not n_last.empty:
            l = n_last.iloc[-1]
            st.metric("Nowcast primario 12m (%PIB)", f"{float(l[n_cols[0]]):.2f}%")
            if len(n_cols) > 1:
                st.metric("Nowcast financiero 12m (%PIB)", f"{float(l[n_cols[1]]):.2f}%")
        st.caption(
            "Nowcast experimental: fiscal acumulado 12m / PIB nominal 12m estimado (calibrado con PIB trimestral oficial y extendido con EMAE+IPC)."
        )
        st.caption(f"Fuente nowcast: {payload['fiscal_nowcast'].source or 'n/d'}")

_explain(
    "Balance fiscal en dos lentes: historico anual (%PBI) y pulso mensual reciente (niveles y acumulado 12m).",
    "El anual muestra sostenibilidad de mediano plazo; el mensual evita quedar ciego cuando PBI publica con rezago.",
    "Si mejora sostenida en 12m mensual, anticipa mejora del ratio fiscal futuro; deterioro mensual persistente alerta sobre riesgo macro.",
)
st.caption(
    f"Fuentes: {payload['fiscal'].source or 'n/d'} | {payload['fiscal_monthly'].source or 'n/d'}"
)

st.subheader("6) Balanza comercial anual")
trade = _slice_year(payload["trade"].dataframe.copy(), since.year, until.year)
if trade.empty:
    _show_unavailable("No se pudo obtener INDEC balanza comercial.")
else:
    t = trade.reset_index().rename(columns={"Anio": "year"})
    bars = (
        alt.Chart(t)
        .mark_bar(color="#0EA5E9")
        .encode(x=alt.X("year:O", title="Anio"), y=alt.Y("Balanza Comercial:Q", title="Saldo (MM USD)"))
    )
    lines = (
        alt.Chart(t)
        .transform_fold(["Exportaciones", "Importaciones"], as_=["Serie", "Valor"])
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("year:O"),
            y=alt.Y("Valor:Q", title="Expo/Impo (MM USD)"),
            color=alt.Color("Serie:N", scale=alt.Scale(range=["#16A34A", "#EF4444"])),
        )
    )
    st.altair_chart(alt.layer(bars, lines).resolve_scale(y="independent").properties(height=chart_height), width="stretch")
    _explain(
        "Saldo externo y dinamica de exportaciones/importaciones.",
        "La restriccion externa sigue siendo una variable clave para FX y riesgo país.",
        "Superavit robusto reduce vulnerabilidad cambiaria; deterioro rapido suele anticipar tension en brecha y reservas.",
    )
    st.caption(f"Fuente: {payload['trade'].source or 'n/d'}")

st.subheader("7) Liquidez monetaria oficial (M2 y base monetaria)")
liq = _slice_date(payload["liquidity"].dataframe.copy(), since, until)
if liq.empty:
    _show_unavailable("No se pudo obtener liquidez monetaria oficial.")
else:
    l_cols = [c for c in ["M2_Millones_ARS", "Base_Monetaria_Millones_ARS"] if c in liq.columns]
    if l_cols:
        st.line_chart(liq[l_cols], height=chart_height, width="stretch")
    yoy_cols = [c for c in ["M2_YoY_Pct", "M2_Real_YoY_Pct"] if c in liq.columns]
    if yoy_cols:
        st.line_chart(liq[yoy_cols].dropna(how="all"), height=280, width="stretch")
    _explain(
        "Evolucion de agregados monetarios y crecimiento real del M2.",
        "Es una medida directa de impulso monetario en una economia inflacionaria.",
        "M2 real en aceleracion suele aumentar sensibilidad a nominalidad y cobertura; desaceleracion real apoya desinflacion.",
    )
    st.caption(f"Fuente: {payload['liquidity'].source or 'n/d'}")

st.subheader("8) Tasas reales (ex-post y ex-ante)")
rates = _slice_date(payload["rates"].dataframe.copy(), since, until)
if rates.empty:
    _show_unavailable("No se pudo construir panel de tasas reales.")
else:
    r_cols = [c for c in ["Tasa_Real_ExPost_Pct", "Tasa_Real_ExAnte_Pct"] if c in rates.columns]
    if r_cols:
        st.line_chart(rates[r_cols].dropna(how="all"), height=chart_height, width="stretch")
    if "Badlar_TNA_Pct" in rates.columns and "REM_Inflacion_12m_Pct" in rates.columns:
        latest = rates[["Badlar_TNA_Pct", "REM_Inflacion_12m_Pct"]].dropna().iloc[-1]
        st.metric("Carry real ex-ante (Badlar - REM)", f"{(latest['Badlar_TNA_Pct'] - latest['REM_Inflacion_12m_Pct']):.2f} pp")
    _explain(
        "Retorno real de instrumentos en pesos contra inflacion observada y esperada.",
        "Define atractivo relativo de carry en pesos vs cobertura dolar.",
        "Tasa real positiva y estable favorece duration en pesos; tasa real negativa sostenida sube incentivo a dolarizar cartera.",
    )
    st.caption(f"Fuente: {payload['rates'].source or 'n/d'}")

st.subheader("9) Soja real (USD por tonelada, deflactada por CPI EEUU)")
soy = _slice_date(payload["soy"].dataframe.copy(), since, until)
if soy.empty or "Soja_Real" not in soy.columns:
    _show_unavailable("No se pudo obtener soja ajustada.")
else:
    st.line_chart(soy[["Soja_Real"]].dropna(), height=chart_height, width="stretch")
    _explain(
        "Precio real de soja como termometro de terminos de intercambio para Argentina.",
        "Impacta ingreso de divisas, recaudacion y balance externo.",
        "Suba sostenida mejora panorama externo y activos locales; baja prolongada deteriora colchones macro.",
    )
    st.caption(f"Fuente: {payload['soy'].source or 'n/d'}")

st.subheader("10) Semaforo macro de decision")
score_rows: list[dict[str, float | str]] = []

infl_df = payload["infl_exp"].dataframe
if not infl_df.empty and "Inflacion_Interanual_Pct" in infl_df:
    val = float(infl_df["Inflacion_Interanual_Pct"].dropna().iloc[-1])
    score_rows.append({"factor": "Inflacion interanual", "valor": val, "score": max(0.0, min(100.0, 100 - val))})

rates_df = payload["rates"].dataframe
if not rates_df.empty and "Tasa_Real_ExAnte_Pct" in rates_df:
    val = float(rates_df["Tasa_Real_ExAnte_Pct"].dropna().iloc[-1])
    score_rows.append({"factor": "Tasa real ex-ante", "valor": val, "score": max(0.0, min(100.0, 50 + val * 4))})

liq_df = payload["liquidity"].dataframe
if not liq_df.empty and "M2_Real_YoY_Pct" in liq_df:
    val = float(liq_df["M2_Real_YoY_Pct"].dropna().iloc[-1])
    score_rows.append({"factor": "M2 real YoY", "valor": val, "score": max(0.0, min(100.0, 60 - val * 2))})

if score_rows:
    s = pd.DataFrame(score_rows)
    score = float(s["score"].mean())
    st.metric("Indice de condiciones macro (0-100)", f"{score:.1f}")
    bars = (
        alt.Chart(s)
        .mark_bar(color="#2563EB")
        .encode(x=alt.X("factor:N", title="Factor"), y=alt.Y("score:Q", title="Score"), tooltip=["factor", "valor", "score"])
        .properties(height=280)
    )
    st.altair_chart(bars, width="stretch")
else:
    _show_unavailable("No hay suficientes series para construir semaforo macro.")

_explain(
    "Indicador sintesis de inflacion, tasa real y liquidez real.",
    "Resume el estado macro para decisiones tacticas de exposicion local.",
    "No reemplaza analisis discrecional; sirve para detectar cambios de regimen mas rapido.",
)

with st.expander("Eventos historicos considerados"):
    st.dataframe(markers.rename(columns={"fecha": "Fecha", "evento": "Evento"}), width="stretch")

st.divider()
st.caption("Diagnostico de fuentes y frescura")
diag_rows = []
for key, result in payload.items():
    diag_rows.append(
        {
            "panel": key,
            "status": result.status,
            "filas": int(len(result.dataframe)),
            "ultimo_dato": str(result.last_date or "n/d"),
            "fuente_o_error": result.source or result.notes or "n/d",
        }
    )
st.dataframe(pd.DataFrame(diag_rows), width="stretch")
