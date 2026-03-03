from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
import sys
from io import StringIO
import warnings

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_analyzer.data.macro import get_indec_trade_balance


@dataclass
class SeriesResult:
    dataframe: pd.DataFrame
    source: str
    status: str = "ok"
    last_date: date | None = None
    notes: str = ""


ARGENTINA_DATOS_CANDIDATES = {
    "dolares": [
        "https://api.argentinadatos.com/v1/cotizaciones/dolares/",
        "https://api.argentinadatos.com/v1/cotizaciones/dolares",
    ],
    "ipc": [
        "https://api.argentinadatos.com/v1/finanzas/indices/inflacion",
    ],
    "m2_palermo": [
        "https://api.argentinadatos.com/v1/inmuebles/precio-m2",
        "https://api.argentinadatos.com/v1/inmuebles/m2",
    ],
}

SERIES_API_URL = "https://apis.datos.gob.ar/series/api/series"
BCRA_V4_BASE_URL = "https://api.bcra.gob.ar/estadisticas/v4.0/monetarias"
SERIES_IDS = {
    "emae_desest": "143.3_NO_PR_2004_A_31",
    "salario_registrado": "149.1_TL_REGIADO_OCTU_0_16",
    "itcrm_mensual": "116.3_TCRMA_0_M_36",
    "fiscal_primario_mensual": "452.3_RESULTADO_RIO_0_M_18_54",
    "fiscal_financiero_mensual": "452.3_RESULTADO_ERO_0_M_20_25",
    "pib_anual": "pib_serie",
    "pib_trimestral": "166.2_PPIB_0_0_3",
}

BCRA_V4_SERIES_IDS = {
    "inflacion_mensual": 27,
    "inflacion_interanual": 28,
    "rem_inflacion_12m": 29,
    "badlar_priv_tna": 7,
    "m2": 109,
    "base_monetaria": 71,
}

FALLBACK_SERIES_IDS = {
    "rem_12m_median": "430.1_MEDIANA_IP_12_M_0_0_27_96",
    "m2_monthly": "174.1_AGADOS_M2_0_0_28",
    "base_monetaria_monthly": "90.1_BMCCB_0_0_36",
    "badlar_daily": "89.2_TS_INTELAR_0_D_20",
}


def _safe_get_json(url: str, timeout: int = 25) -> list[dict] | None:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "results", "resultados"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return None


def _safe_get_payload(url: str, timeout: int = 25) -> object | None:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _index_last_date(df: pd.DataFrame) -> date | None:
    if df.empty:
        return None
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
        return idx.max().date()
    if pd.api.types.is_integer_dtype(idx):
        year = int(pd.Series(idx).dropna().max())
        if 1900 <= year <= 2200:
            return date(year, 12, 31)
    if len(idx) == 0:
        return None
    last = idx.max()
    if isinstance(last, str) and last.isdigit() and len(last) == 4:
        year = int(last)
        if 1900 <= year <= 2200:
            return date(year, 12, 31)
    if isinstance(last, (int, float)):
        year = int(last)
        if 1900 <= year <= 2200:
            return date(year, 12, 31)
        return None
    try:
        parsed = pd.to_datetime(last, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.date()


def _mark_freshness(
    df: pd.DataFrame,
    source: str,
    stale_after_days: int,
    notes: str = "",
) -> SeriesResult:
    last = _index_last_date(df)
    if last is None:
        return SeriesResult(dataframe=df, source=source, status="error", notes=notes or "Serie vacia")
    age = (pd.Timestamp.today().date() - last).days
    status = "ok" if age <= stale_after_days else "stale"
    return SeriesResult(dataframe=df, source=source, status=status, last_date=last, notes=notes)


def _get_bcra_v4_series(variable_id: int, limit: int = 6000) -> pd.Series:
    url = f"{BCRA_V4_BASE_URL}/{variable_id}"
    params = {"limit": limit}
    headers = {"user-agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, params=params, timeout=30, headers=headers)
        response.raise_for_status()
    except requests.exceptions.SSLError:
        try:
            warnings.filterwarnings("ignore", message="Unverified HTTPS request")
            response = requests.get(url, params=params, timeout=30, headers=headers, verify=False)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            return pd.Series(dtype=float)
    except requests.exceptions.RequestException:
        return pd.Series(dtype=float)
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        return pd.Series(dtype=float)
    detail = results[0].get("detalle", [])
    if not detail:
        return pd.Series(dtype=float)
    raw = pd.DataFrame(detail)
    if "fecha" not in raw.columns or "valor" not in raw.columns:
        return pd.Series(dtype=float)
    raw["fecha"] = pd.to_datetime(raw["fecha"], errors="coerce")
    raw["valor"] = pd.to_numeric(raw["valor"], errors="coerce")
    raw = raw.dropna(subset=["fecha", "valor"]).sort_values("fecha")
    if raw.empty:
        return pd.Series(dtype=float)
    return raw.set_index("fecha")["valor"].astype(float)


def _pick_first_dataframe(urls: Iterable[str]) -> tuple[pd.DataFrame, str]:
    for url in urls:
        payload = _safe_get_payload(url)
        if payload is None:
            continue
        if isinstance(payload, list):
            df = pd.DataFrame(payload)
            if not df.empty:
                return df, url
            continue
        if isinstance(payload, dict):
            for key in ("data", "results", "resultados"):
                value = payload.get(key)
                if isinstance(value, list):
                    df = pd.DataFrame(value)
                    if not df.empty:
                        return df, url
    return pd.DataFrame(), ""


def _find_column(columns: Iterable[str], options: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for option in options:
        for key, original in lower.items():
            if option in key:
                return original
    return None


def _to_monthly_series(
    raw_df: pd.DataFrame,
    date_candidates: tuple[str, ...],
    value_candidates: tuple[str, ...],
    name: str,
) -> pd.Series:
    if raw_df.empty:
        return pd.Series(dtype=float, name=name)

    date_col = _find_column(raw_df.columns, date_candidates)
    value_col = _find_column(raw_df.columns, value_candidates)
    if date_col is None or value_col is None:
        return pd.Series(dtype=float, name=name)

    out = raw_df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if out.empty:
        return pd.Series(dtype=float, name=name)

    out["month"] = out[date_col].dt.to_period("M").dt.to_timestamp()
    series = out.groupby("month")[value_col].mean().rename(name)
    return series.astype(float)


def _rebase(series: pd.Series, base_date: str) -> pd.Series:
    if series.empty:
        return series
    base_ts = pd.Timestamp(base_date).to_period("M").to_timestamp()
    base = series.loc[series.index <= base_ts]
    if base.empty:
        return series / series.iloc[0]
    base_val = float(base.iloc[-1])
    if base_val == 0:
        return series
    return series / base_val


def _get_series_api(ids: list[str], start_date: str | None = None) -> pd.DataFrame:
    # API can truncate long daily histories; request newest window then sort ascending locally.
    params = {"ids": ",".join(ids), "limit": 5000, "sort": "desc"}
    if start_date:
        params["start_date"] = start_date
    response = requests.get(
        SERIES_API_URL,
        params=params,
        timeout=30,
        headers={"user-agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame(columns=ids)
    cols = ["fecha"] + ids
    df = pd.DataFrame(rows, columns=cols)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    for col in ids:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["fecha"]).set_index("fecha").sort_index()
    return df


def _safe_get_series_api(ids: list[str], start_date: str | None = None) -> pd.DataFrame:
    try:
        return _get_series_api(ids, start_date=start_date)
    except Exception:
        return pd.DataFrame(columns=ids)


def _normalize_pct_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return series
    median_abs = float(clean.abs().median())
    # Some official feeds provide percentage rates as fractions (e.g. 0.21 for 21%).
    if 0 < median_abs < 2:
        return series * 100.0
    return series


def _get_dolar_casa_series(casa: str, name: str) -> tuple[pd.Series, str]:
    raw, source = _pick_first_dataframe(ARGENTINA_DATOS_CANDIDATES["dolares"])
    if raw.empty:
        return pd.Series(dtype=float, name=name), source
    if "casa" not in raw.columns or "fecha" not in raw.columns:
        return pd.Series(dtype=float, name=name), source
    value_col = "venta" if "venta" in raw.columns else "valor"
    if value_col not in raw.columns:
        return pd.Series(dtype=float, name=name), source

    out = raw.copy()
    out["casa"] = out["casa"].astype(str).str.lower()
    out = out[out["casa"] == casa.lower()]
    out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["fecha", value_col]).sort_values("fecha")
    if out.empty:
        return pd.Series(dtype=float, name=name), source
    series = (
        out.groupby(out["fecha"].dt.to_period("M").dt.to_timestamp())[value_col]
        .mean()
        .rename(name)
    )
    return series.astype(float), source


def _get_inflation_monthly() -> tuple[pd.Series, str]:
    raw, source = _pick_first_dataframe(ARGENTINA_DATOS_CANDIDATES["ipc"])
    if raw.empty:
        return pd.Series(dtype=float, name="Inflacion_mensual"), source
    series = _to_monthly_series(
        raw,
        date_candidates=("fecha", "date", "periodo"),
        value_candidates=("valor", "inflacion", "value"),
        name="Inflacion_mensual",
    )
    return series, source


def get_emae_series() -> SeriesResult:
    df = _get_series_api([SERIES_IDS["emae_desest"]])
    if df.empty:
        return SeriesResult(dataframe=pd.DataFrame(), source="datos.gob.ar series API", status="error")
    out = df.rename(columns={SERIES_IDS["emae_desest"]: "EMAE"})[["EMAE"]]
    return _mark_freshness(out, "datos.gob.ar series API", stale_after_days=90)


def get_ccl_inflation_wages(base_date: str = "2023-12-01") -> SeriesResult:
    ccl, ccl_source = _get_dolar_casa_series("contadoconliqui", "CCL")
    monthly_infl, ipc_source = _get_inflation_monthly()
    infl_index = (1 + (monthly_infl / 100.0).fillna(0)).cumprod().rename("Inflacion")
    wages_df = _get_series_api([SERIES_IDS["salario_registrado"]])
    wages = wages_df[SERIES_IDS["salario_registrado"]].rename("Salarios") if not wages_df.empty else pd.Series(dtype=float, name="Salarios")

    combined = pd.concat([ccl, infl_index, wages], axis=1).dropna(how="all")
    combined = combined.sort_index()
    if combined.empty:
        return SeriesResult(dataframe=combined, source="argentinadatos + datos.gob.ar", status="error")

    rebased = pd.DataFrame(index=combined.index)
    for col in ("CCL", "Inflacion", "Salarios"):
        if col in combined:
            rebased[col] = _rebase(combined[col].dropna(), base_date=base_date)

    return _mark_freshness(
        rebased,
        " | ".join(filter(None, [ccl_source, ipc_source, "datos.gob.ar salarios"])),
        stale_after_days=60,
    )


def get_fx_vs_itcrm(base_date: str = "2026-02-01") -> SeriesResult:
    official, off_source = _get_dolar_casa_series("oficial", "Dolar_Oficial")
    ccl, ccl_source = _get_dolar_casa_series("contadoconliqui", "Dolar_CCL")
    itcrm_df = _get_series_api([SERIES_IDS["itcrm_mensual"]])
    itcrm = (
        itcrm_df[SERIES_IDS["itcrm_mensual"]].rename("ITCRM")
        if not itcrm_df.empty
        else pd.Series(dtype=float, name="ITCRM")
    )

    combined = pd.concat([official, ccl, itcrm], axis=1).dropna(how="all").sort_index()
    if combined.empty:
        return SeriesResult(dataframe=combined, source="argentinadatos + datos.gob.ar", status="error")

    scale = _rebase(combined["ITCRM"].dropna(), base_date=base_date)
    if "Dolar_Oficial" in combined:
        combined["Oficial_Ajustado_ITCRM"] = (
            combined["Dolar_Oficial"] / scale.reindex(combined.index).ffill()
        )
    if "Dolar_CCL" in combined:
        combined["CCL_Ajustado_ITCRM"] = combined["Dolar_CCL"] / scale.reindex(combined.index).ffill()
    if "Oficial_Ajustado_ITCRM" in combined:
        combined["Oficial_Ajustado_ITCRM_Base"] = _rebase(
            combined["Oficial_Ajustado_ITCRM"].dropna(), base_date=base_date
        ).reindex(combined.index)
    if "CCL_Ajustado_ITCRM" in combined:
        combined["CCL_Ajustado_ITCRM_Base"] = _rebase(
            combined["CCL_Ajustado_ITCRM"].dropna(), base_date=base_date
        ).reindex(combined.index)

    return _mark_freshness(
        combined,
        " | ".join(filter(None, [off_source, ccl_source, "datos.gob.ar ITCRM"])),
        stale_after_days=60,
    )


def get_fiscal_result_pct_gdp() -> SeriesResult:
    repo_root = Path(__file__).resolve().parents[3]
    local_csv = repo_root / "data" / "fiscal_resultado_pct_pbi.csv"
    if local_csv.exists():
        local_df = pd.read_csv(local_csv)
        source = str(local_csv)
    else:
        try:
            fiscal = _get_series_api(
                [SERIES_IDS["fiscal_primario_mensual"], SERIES_IDS["fiscal_financiero_mensual"]],
                start_date="2016-01-01",
            )
            pib = _get_series_api([SERIES_IDS["pib_anual"]], start_date="2016-01-01")
            pib_q = _get_series_api([SERIES_IDS["pib_trimestral"]], start_date="2016-01-01")
        except Exception:
            return SeriesResult(dataframe=pd.DataFrame(), source="datos.gob.ar IMIG/PIB", status="error")

        if fiscal.empty or pib.empty:
            return SeriesResult(dataframe=pd.DataFrame(), source="datos.gob.ar IMIG/PIB", status="error")

        annual_fiscal = fiscal.resample("YE").sum(numeric_only=True)
        annual_fiscal.index = annual_fiscal.index.year
        annual_fiscal.index.name = "Anio"

        pib_annual = pib.copy()
        pib_annual.index = pib_annual.index.year
        pib_annual.index.name = "Anio"
        pib_annual = pib_annual.groupby(level=0).last()
        # pib_serie is in ARS; convert to million ARS to match fiscal series units.
        pib_annual[SERIES_IDS["pib_anual"]] = pib_annual[SERIES_IDS["pib_anual"]] / 1_000_000

        # Extend annual GDP with quarterly data when full 4 quarters are available.
        if not pib_q.empty:
            tmp = pib_q.copy()
            tmp["Anio"] = tmp.index.year
            quarterly_counts = tmp.groupby("Anio")[SERIES_IDS["pib_trimestral"]].count()
            quarterly_sums = tmp.groupby("Anio")[SERIES_IDS["pib_trimestral"]].sum()
            extension = quarterly_sums[quarterly_counts >= 4]
            extension = extension[extension.index > int(pib_annual.index.max())]
            if not extension.empty:
                ext_df = extension.to_frame(name=SERIES_IDS["pib_anual"])
                ext_df.index.name = "Anio"
                pib_annual = pd.concat([pib_annual, ext_df], axis=0)

        joined = annual_fiscal.join(pib_annual, how="inner")
        if joined.empty:
            return SeriesResult(dataframe=pd.DataFrame(), source="datos.gob.ar IMIG/PIB", status="error")

        result = pd.DataFrame(index=joined.index)
        result.index.name = "Anio"
        # Both fiscal and PIB are expressed in million ARS at this point.
        result["Resultado_Primario_Pct_PBI"] = (
            joined[SERIES_IDS["fiscal_primario_mensual"]] / joined[SERIES_IDS["pib_anual"]]
        ) * 100
        result["Resultado_Financiero_Pct_PBI"] = (
            joined[SERIES_IDS["fiscal_financiero_mensual"]] / joined[SERIES_IDS["pib_anual"]]
        ) * 100
        return _mark_freshness(
            result.dropna(how="all"),
            "datos.gob.ar IMIG + PIB (anual + trimestral)",
            stale_after_days=480,
        )

    if local_df.empty:
        return SeriesResult(dataframe=pd.DataFrame(), source=source, status="error")

    year_col = _find_column(local_df.columns, ("anio", "ano", "year", "fecha"))
    primary_col = _find_column(local_df.columns, ("primario", "resultado_primario", "rdo_primario"))
    fin_col = _find_column(local_df.columns, ("financiero", "resultado_financiero", "rdo_financiero"))
    gdp_col = _find_column(local_df.columns, ("pbi", "gdp"))

    if year_col is None:
        return SeriesResult(dataframe=pd.DataFrame(), source=source, status="error")

    out = local_df.copy()
    if "fecha" in year_col.lower():
        out["anio"] = pd.to_datetime(out[year_col], errors="coerce").dt.year
    else:
        out["anio"] = pd.to_numeric(out[year_col], errors="coerce")
    out = out.dropna(subset=["anio"]).copy()
    out["anio"] = out["anio"].astype(int)

    for col in (primary_col, fin_col, gdp_col):
        if col:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    result = pd.DataFrame(index=sorted(out["anio"].unique()))
    result.index.name = "Anio"

    if primary_col:
        result["Resultado_Primario_Pct_PBI"] = out.groupby("anio")[primary_col].mean()
    if fin_col:
        result["Resultado_Financiero_Pct_PBI"] = out.groupby("anio")[fin_col].mean()

    if gdp_col and primary_col and "Resultado_Primario_Pct_PBI" not in result:
        annual = out.groupby("anio")[[primary_col, gdp_col]].sum()
        result["Resultado_Primario_Pct_PBI"] = (annual[primary_col] / annual[gdp_col]) * 100
    if gdp_col and fin_col and "Resultado_Financiero_Pct_PBI" not in result:
        annual = out.groupby("anio")[[fin_col, gdp_col]].sum()
        result["Resultado_Financiero_Pct_PBI"] = (annual[fin_col] / annual[gdp_col]) * 100

    return _mark_freshness(result.dropna(how="all"), source=source, stale_after_days=480)


def get_fiscal_monthly_result() -> SeriesResult:
    fiscal = _safe_get_series_api(
        [SERIES_IDS["fiscal_primario_mensual"], SERIES_IDS["fiscal_financiero_mensual"]],
        start_date="2016-01-01",
    )
    if fiscal.empty:
        return SeriesResult(
            dataframe=pd.DataFrame(),
            source="datos.gob.ar IMIG",
            status="error",
            notes="Sin serie mensual fiscal",
        )

    out = pd.DataFrame(index=fiscal.index)
    out["Resultado_Primario_Mill_ARS"] = pd.to_numeric(
        fiscal.get(SERIES_IDS["fiscal_primario_mensual"]), errors="coerce"
    )
    out["Resultado_Financiero_Mill_ARS"] = pd.to_numeric(
        fiscal.get(SERIES_IDS["fiscal_financiero_mensual"]), errors="coerce"
    )
    out = out.sort_index().dropna(how="all")
    if out.empty:
        return SeriesResult(
            dataframe=out,
            source="datos.gob.ar IMIG",
            status="error",
            notes="Serie mensual vacia",
        )

    out["Resultado_Primario_12m_Mill_ARS"] = out["Resultado_Primario_Mill_ARS"].rolling(12).sum()
    out["Resultado_Financiero_12m_Mill_ARS"] = out["Resultado_Financiero_Mill_ARS"].rolling(12).sum()
    return _mark_freshness(out, source="datos.gob.ar IMIG", stale_after_days=60)


def get_fiscal_nowcast_pct_gdp() -> SeriesResult:
    fiscal = get_fiscal_monthly_result().dataframe.copy()
    if fiscal.empty:
        return SeriesResult(
            dataframe=pd.DataFrame(),
            source="datos.gob.ar IMIG + nowcast PIB",
            status="error",
            notes="Sin fiscal mensual para construir nowcast",
        )

    pib_q = _safe_get_series_api([SERIES_IDS["pib_trimestral"]], start_date="2016-01-01")
    emae_df = _safe_get_series_api([SERIES_IDS["emae_desest"]], start_date="2016-01-01")
    infl_m, infl_source = _get_inflation_monthly()

    if pib_q.empty or emae_df.empty or infl_m.empty:
        return SeriesResult(
            dataframe=pd.DataFrame(),
            source="datos.gob.ar PIB trimestral + EMAE + IPC",
            status="error",
            notes="Faltan insumos para nowcast fiscal %PIB",
        )

    pib_q_series = pd.to_numeric(pib_q.get(SERIES_IDS["pib_trimestral"]), errors="coerce").dropna().sort_index()
    if pib_q_series.empty:
        return SeriesResult(
            dataframe=pd.DataFrame(),
            source="datos.gob.ar PIB trimestral + EMAE + IPC",
            status="error",
            notes="PIB trimestral vacio",
        )

    # Convert quarter GDP to monthly average level for calibration.
    pib_q_to_m = pib_q_series.resample("MS").ffill() / 3.0

    emae = pd.to_numeric(emae_df.get(SERIES_IDS["emae_desest"]), errors="coerce").dropna().sort_index()
    emae = emae.resample("MS").mean()
    infl_idx = (1 + (infl_m / 100.0).fillna(0)).cumprod().resample("MS").last()

    nominal_proxy = (emae * infl_idx).rename("Nominal_Proxy")
    overlap = pd.concat([pib_q_to_m.rename("PIB_Mensual_Oficial"), nominal_proxy], axis=1).dropna()
    if overlap.empty:
        return SeriesResult(
            dataframe=pd.DataFrame(),
            source="datos.gob.ar PIB trimestral + EMAE + IPC",
            status="error",
            notes="Sin overlap para calibrar nowcast",
        )

    # Robust scale from most recent overlap window.
    ratio = (overlap["PIB_Mensual_Oficial"] / overlap["Nominal_Proxy"]).replace([pd.NA, pd.NaT], pd.NA).dropna()
    ratio_tail = ratio.tail(12)
    scale = float(ratio_tail.median()) if not ratio_tail.empty else float(ratio.median())
    if not pd.notna(scale) or scale <= 0:
        return SeriesResult(
            dataframe=pd.DataFrame(),
            source="datos.gob.ar PIB trimestral + EMAE + IPC",
            status="error",
            notes="Escala invalida en nowcast PIB",
        )

    pib_nowcast_m = (nominal_proxy * scale).rename("PIB_Mensual_Nowcast")
    # Prefer official monthlyized quarter GDP where available.
    pib_m_final = pib_nowcast_m.combine_first(pib_q_to_m).sort_index()
    pib_m_final = pib_q_to_m.combine_first(pib_m_final).sort_index()
    pib_m_final = pib_m_final.rename("PIB_Mensual_Nowcast")

    out = fiscal.copy().sort_index()
    out = out.join(pib_m_final, how="left")
    out["PIB_12m_Nowcast_Mill_ARS"] = out["PIB_Mensual_Nowcast"].rolling(12).sum()
    out["Resultado_Primario_12m_Pct_PIB_Nowcast"] = (
        out["Resultado_Primario_12m_Mill_ARS"] / out["PIB_12m_Nowcast_Mill_ARS"]
    ) * 100.0
    out["Resultado_Financiero_12m_Pct_PIB_Nowcast"] = (
        out["Resultado_Financiero_12m_Mill_ARS"] / out["PIB_12m_Nowcast_Mill_ARS"]
    ) * 100.0
    out = out.dropna(
        subset=["Resultado_Primario_12m_Pct_PIB_Nowcast", "Resultado_Financiero_12m_Pct_PIB_Nowcast"],
        how="all",
    )
    return _mark_freshness(
        out,
        source=f"datos.gob.ar IMIG + PIB trimestral + EMAE + IPC ({infl_source or 'argentinadatos'})",
        stale_after_days=60,
        notes="Nowcast experimental de %PIB con calibracion mensual al PIB trimestral",
    )


def get_salary_index(base_date: str = "2026-02-01") -> SeriesResult:
    df = _get_series_api([SERIES_IDS["salario_registrado"]])
    if df.empty:
        return SeriesResult(dataframe=pd.DataFrame(), source="datos.gob.ar series API", status="error")
    out = df.rename(columns={SERIES_IDS["salario_registrado"]: "Indice_Salarios"})[["Indice_Salarios"]]
    infl_m, _ = _get_inflation_monthly()
    if infl_m.empty:
        out["Indice_Salarios_Nominal_Base"] = _rebase(out["Indice_Salarios"], base_date=base_date)
        out["Indice_Salarios_Real_Base"] = out["Indice_Salarios_Nominal_Base"]
        return _mark_freshness(out, "datos.gob.ar series API (fallback nominal)", stale_after_days=90)

    infl_idx = (1 + (infl_m / 100.0).fillna(0)).cumprod().rename("IPC")
    infl_aligned = infl_idx.reindex(out.index).ffill()
    out["Indice_Salarios_Real"] = out["Indice_Salarios"] / infl_aligned
    out["Indice_Salarios_Nominal_Base"] = _rebase(out["Indice_Salarios"], base_date=base_date)
    out["Indice_Salarios_Real_Base"] = _rebase(out["Indice_Salarios_Real"], base_date=base_date)
    return _mark_freshness(out, "datos.gob.ar series API", stale_after_days=90)


def get_trade_balance_annual() -> SeriesResult:
    result = get_indec_trade_balance()
    df = result.dataframe.copy()
    annual = df.resample("YE").sum(numeric_only=True)
    counts = df["Balanza Comercial"].resample("YE").count()
    annual = annual[counts >= 12]
    annual.index = annual.index.year
    annual.index.name = "Anio"
    return _mark_freshness(annual, result.source_url, stale_after_days=480)


def get_inflation_and_expectations() -> SeriesResult:
    source_parts: list[str] = []
    monthly = _get_bcra_v4_series(BCRA_V4_SERIES_IDS["inflacion_mensual"]).rename("Inflacion_Mensual_Pct")
    yoy = _get_bcra_v4_series(BCRA_V4_SERIES_IDS["inflacion_interanual"]).rename("Inflacion_Interanual_Pct")
    rem = _get_bcra_v4_series(BCRA_V4_SERIES_IDS["rem_inflacion_12m"]).rename("REM_Inflacion_12m_Pct")

    if not monthly.empty:
        source_parts.append("BCRA API v4 IPC mensual")
    else:
        monthly_fallback, fallback_source = _get_inflation_monthly()
        monthly = monthly_fallback.rename("Inflacion_Mensual_Pct")
        if not monthly.empty:
            source_parts.append(fallback_source or "argentinadatos IPC")

    if yoy.empty and not monthly.empty:
        yoy = (((1 + monthly / 100.0).rolling(12).apply(lambda x: x.prod(), raw=True)) - 1) * 100
        yoy = yoy.rename("Inflacion_Interanual_Pct")
        source_parts.append("Inflacion interanual calculada desde mensual")
    elif not yoy.empty:
        source_parts.append("BCRA API v4 IPC interanual")

    if rem.empty:
        rem_df = _safe_get_series_api([FALLBACK_SERIES_IDS["rem_12m_median"]])
        if not rem_df.empty and FALLBACK_SERIES_IDS["rem_12m_median"] in rem_df.columns:
            rem = _normalize_pct_series(rem_df[FALLBACK_SERIES_IDS["rem_12m_median"]]).rename(
                "REM_Inflacion_12m_Pct"
            )
            source_parts.append("datos.gob.ar REM mediana 12m")
    else:
        source_parts.append("BCRA API v4 REM 12m")

    out = pd.concat([monthly, yoy, rem], axis=1).sort_index()
    return _mark_freshness(out, source=" | ".join(source_parts), stale_after_days=45)


def get_monetary_liquidity() -> SeriesResult:
    source_parts: list[str] = []
    m2 = _get_bcra_v4_series(BCRA_V4_SERIES_IDS["m2"]).rename("M2_Millones_ARS")
    base = _get_bcra_v4_series(BCRA_V4_SERIES_IDS["base_monetaria"]).rename("Base_Monetaria_Millones_ARS")
    infl = _get_bcra_v4_series(BCRA_V4_SERIES_IDS["inflacion_mensual"]).rename("Inflacion_Mensual_Pct")

    if m2.empty:
        m2_df = _safe_get_series_api([FALLBACK_SERIES_IDS["m2_monthly"]])
        if not m2_df.empty and FALLBACK_SERIES_IDS["m2_monthly"] in m2_df.columns:
            m2 = m2_df[FALLBACK_SERIES_IDS["m2_monthly"]].rename("M2_Millones_ARS")
            source_parts.append("datos.gob.ar M2 mensual")
    else:
        source_parts.append("BCRA API v4 M2")

    if base.empty:
        base_df = _safe_get_series_api([FALLBACK_SERIES_IDS["base_monetaria_monthly"]])
        if not base_df.empty and FALLBACK_SERIES_IDS["base_monetaria_monthly"] in base_df.columns:
            base = base_df[FALLBACK_SERIES_IDS["base_monetaria_monthly"]].rename("Base_Monetaria_Millones_ARS")
            source_parts.append("datos.gob.ar Base Monetaria")
    else:
        source_parts.append("BCRA API v4 Base Monetaria")

    if infl.empty:
        infl_fallback, infl_source = _get_inflation_monthly()
        infl = infl_fallback.rename("Inflacion_Mensual_Pct")
        if not infl.empty:
            source_parts.append(infl_source or "argentinadatos IPC")
    else:
        source_parts.append("BCRA API v4 IPC mensual")

    out = pd.concat([m2, base], axis=1).sort_index()
    if out.empty:
        return SeriesResult(
            dataframe=out,
            source="BCRA API v4 + datos.gob.ar",
            status="error",
            notes="Sin datos de liquidez",
        )

    monthly = out.resample("MS").mean().dropna(how="all")
    monthly["M2_YoY_Pct"] = monthly["M2_Millones_ARS"].pct_change(12) * 100

    if not infl.empty:
        infl_m = infl.resample("MS").last().reindex(monthly.index).ffill()
        cpi_idx = (1 + (infl_m / 100.0).fillna(0)).cumprod()
        monthly["M2_Real_Index"] = monthly["M2_Millones_ARS"] / cpi_idx
        monthly["M2_Real_YoY_Pct"] = monthly["M2_Real_Index"].pct_change(12) * 100

    return _mark_freshness(
        monthly,
        source=" | ".join(source_parts),
        stale_after_days=45,
    )


def get_real_rates_panel() -> SeriesResult:
    source_parts: list[str] = []
    badlar = _get_bcra_v4_series(BCRA_V4_SERIES_IDS["badlar_priv_tna"]).rename("Badlar_TNA_Pct")
    infl_m = _get_bcra_v4_series(BCRA_V4_SERIES_IDS["inflacion_mensual"]).rename("Inflacion_Mensual_Pct")
    rem = _get_bcra_v4_series(BCRA_V4_SERIES_IDS["rem_inflacion_12m"]).rename("REM_Inflacion_12m_Pct")

    if badlar.empty:
        badlar_df = _safe_get_series_api([FALLBACK_SERIES_IDS["badlar_daily"]])
        if not badlar_df.empty and FALLBACK_SERIES_IDS["badlar_daily"] in badlar_df.columns:
            badlar = badlar_df[FALLBACK_SERIES_IDS["badlar_daily"]].rename("Badlar_TNA_Pct")
            source_parts.append("datos.gob.ar BADLAR")
    else:
        source_parts.append("BCRA API v4 BADLAR")

    if infl_m.empty:
        infl_fallback, infl_source = _get_inflation_monthly()
        infl_m = infl_fallback.rename("Inflacion_Mensual_Pct")
        if not infl_m.empty:
            source_parts.append(infl_source or "argentinadatos IPC")
    else:
        source_parts.append("BCRA API v4 IPC mensual")

    if rem.empty:
        rem_df = _safe_get_series_api([FALLBACK_SERIES_IDS["rem_12m_median"]])
        if not rem_df.empty and FALLBACK_SERIES_IDS["rem_12m_median"] in rem_df.columns:
            rem = _normalize_pct_series(rem_df[FALLBACK_SERIES_IDS["rem_12m_median"]]).rename(
                "REM_Inflacion_12m_Pct"
            )
            source_parts.append("datos.gob.ar REM mediana 12m")
    else:
        source_parts.append("BCRA API v4 REM 12m")

    if badlar.empty:
        return SeriesResult(
            dataframe=pd.DataFrame(),
            source="BCRA API v4 + datos.gob.ar",
            status="error",
            notes="Sin datos BADLAR para tasa real",
        )

    monthly_badlar = badlar.resample("MS").mean().rename("Badlar_TNA_Pct")
    out = monthly_badlar.to_frame()

    if not infl_m.empty:
        infl_m = infl_m.resample("MS").last().reindex(out.index).ffill()
        out["Inflacion_Anualizada_ExPost_Pct"] = ((1 + infl_m / 100.0) ** 12 - 1) * 100
        out["Tasa_Real_ExPost_Pct"] = out["Badlar_TNA_Pct"] - out["Inflacion_Anualizada_ExPost_Pct"]

    if not rem.empty:
        rem = rem.resample("MS").last().reindex(out.index).ffill()
        out["REM_Inflacion_12m_Pct"] = rem
        out["Tasa_Real_ExAnte_Pct"] = out["Badlar_TNA_Pct"] - out["REM_Inflacion_12m_Pct"]

    return _mark_freshness(
        out.dropna(how="all"),
        source=" | ".join(source_parts),
        stale_after_days=45,
    )


def _monthly_inflation(start_date: str = "1990-01-01") -> pd.Series:
    series, _ = _get_inflation_monthly()
    if series.empty:
        return pd.Series(dtype=float)
    return series[series.index >= pd.Timestamp(start_date)]


def _us_cpi_monthly(start_date: str = "1990-01-01") -> pd.Series:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    try:
        response = requests.get(url, timeout=30, headers={"user-agent": "Mozilla/5.0"})
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text))
    except Exception:
        return pd.Series(dtype=float)
    date_col = None
    for candidate in ("DATE", "observation_date", "fecha"):
        if candidate in raw.columns:
            date_col = candidate
            break
    if raw.empty or date_col is None or "CPIAUCSL" not in raw.columns:
        return pd.Series(dtype=float)
    out = raw.copy()
    out["DATE"] = pd.to_datetime(out[date_col], errors="coerce")
    out["CPIAUCSL"] = pd.to_numeric(out["CPIAUCSL"], errors="coerce")
    out = out.dropna(subset=["DATE", "CPIAUCSL"]).set_index("DATE").sort_index()
    out = out["CPIAUCSL"].resample("MS").mean()
    return out[out.index >= pd.Timestamp(start_date)]


def get_soy_real_price(base_date: str = "2026-02-01") -> SeriesResult:
    soy_raw = yf.download("ZS=F", start="1990-01-01", progress=False)
    if isinstance(soy_raw, pd.Series):
        soy = soy_raw
    elif isinstance(soy_raw, pd.DataFrame):
        soy = pd.Series(dtype=float)
        if isinstance(soy_raw.columns, pd.MultiIndex):
            # yfinance can return a multi-indexed DataFrame even for one ticker.
            try:
                close_df = soy_raw.xs("Close", axis=1, level=0)
                if isinstance(close_df, pd.DataFrame) and not close_df.empty:
                    soy = close_df.iloc[:, 0]
            except Exception:
                soy = pd.Series(dtype=float)
        elif "Close" in soy_raw.columns:
            close_col = soy_raw["Close"]
            if isinstance(close_col, pd.Series):
                soy = close_col
            elif isinstance(close_col, pd.DataFrame) and not close_col.empty:
                soy = close_col.iloc[:, 0]
        elif not soy_raw.empty:
            soy = soy_raw.iloc[:, 0]
    else:
        soy = pd.Series(dtype=float)
    if soy.empty:
        return SeriesResult(dataframe=pd.DataFrame(), source="Yahoo Finance", status="error")
    soy = (soy * 0.3674).rename("Soja_USD_ton")
    soy_monthly = soy.resample("MS").mean()

    us_cpi = _us_cpi_monthly(start_date="1990-01-01")
    if us_cpi.empty:
        out = soy_monthly.to_frame()
        out["Soja_Real"] = out["Soja_USD_ton"]
        return _mark_freshness(out, "Yahoo Finance (fallback nominal)", stale_after_days=45)

    base_ts = pd.Timestamp(base_date).to_period("M").to_timestamp()
    cpi_base = us_cpi.loc[us_cpi.index <= base_ts]
    if cpi_base.empty:
        cpi_base_value = float(us_cpi.iloc[-1])
    else:
        cpi_base_value = float(cpi_base.iloc[-1])
    cpi_aligned = us_cpi.reindex(soy_monthly.index).ffill()
    real = (soy_monthly * (cpi_base_value / cpi_aligned)).rename("Soja_Real")

    out = pd.concat([soy_monthly, real], axis=1).dropna(how="all")
    return _mark_freshness(out, "Yahoo Finance | FRED CPIAUCSL", stale_after_days=45)


def get_palermo_m2() -> SeriesResult:
    repo_root = Path(__file__).resolve().parents[3]
    local_csv = repo_root / "data" / "m2_palermo.csv"
    if local_csv.exists():
        df = pd.read_csv(local_csv)
        date_col = _find_column(df.columns, ("fecha", "date", "periodo", "mes"))
        value_col = _find_column(df.columns, ("m2", "precio", "valor", "usd"))
        if date_col and value_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)
            if not df.empty:
                monthly = df.groupby(df[date_col].dt.to_period("M").dt.to_timestamp())[value_col].mean()
                return _mark_freshness(
                    monthly.rename("M2_Palermo").to_frame(),
                    source=str(local_csv),
                    stale_after_days=120,
                    notes="Fuente local experimental",
                )

    raw, source = _pick_first_dataframe(ARGENTINA_DATOS_CANDIDATES["m2_palermo"])
    if raw.empty:
        return SeriesResult(
            dataframe=pd.DataFrame(),
            source=source or "Sin datos. Cargar data/m2_palermo.csv con columnas fecha,m2",
            status="error",
        )

    barrio_col = _find_column(raw.columns, ("barrio", "neighborhood"))
    date_col = _find_column(raw.columns, ("fecha", "date", "periodo"))
    value_col = _find_column(raw.columns, ("m2", "precio", "valor", "price"))

    out = raw.copy()
    if barrio_col:
        palermo_mask = out[barrio_col].astype(str).str.contains("palermo", case=False, na=False)
        if palermo_mask.any():
            out = out[palermo_mask].copy()

    if date_col is None or value_col is None:
        return SeriesResult(dataframe=pd.DataFrame(), source=source, status="error")

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if out.empty:
        return SeriesResult(dataframe=pd.DataFrame(), source=source, status="error")

    monthly = out.groupby(out[date_col].dt.to_period("M").dt.to_timestamp())[value_col].mean()
    monthly = monthly.rename("M2_Palermo")
    return _mark_freshness(monthly.to_frame(), source=source, stale_after_days=120)


def get_macro_regime_markers() -> pd.DataFrame:
    rows = [
        ("2001-12-01", "Crisis 2001 / corralito"),
        ("2002-01-01", "Salida convertibilidad"),
        ("2011-10-01", "Inicio cepo cambiario"),
        ("2015-12-01", "Cambio de regimen cambiario"),
        ("2018-05-01", "Crisis cambiaria / acuerdo FMI"),
        ("2019-08-01", "Shock PASO"),
        ("2020-03-01", "COVID / ASPO"),
        ("2023-12-01", "Cambio de regimen macro"),
    ]
    out = pd.DataFrame(rows, columns=["fecha", "evento"])
    out["fecha"] = pd.to_datetime(out["fecha"], errors="coerce")
    return out.dropna(subset=["fecha"]).sort_values("fecha")
