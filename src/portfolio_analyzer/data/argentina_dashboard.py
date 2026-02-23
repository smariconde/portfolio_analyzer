from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import sys
from io import StringIO

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
SERIES_IDS = {
    "emae_desest": "143.3_NO_PR_2004_A_31",
    "salario_registrado": "149.1_TL_REGIADO_OCTU_0_16",
    "itcrm_mensual": "116.3_TCRMA_0_M_36",
    "fiscal_primario_mensual": "452.3_RESULTADO_RIO_0_M_18_54",
    "fiscal_financiero_mensual": "452.3_RESULTADO_ERO_0_M_20_25",
    "pib_anual": "pib_serie",
    "pib_trimestral": "166.2_PPIB_0_0_3",
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
    params = {"ids": ",".join(ids), "limit": 5000}
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
        return SeriesResult(dataframe=pd.DataFrame(), source="")
    out = df.rename(columns={SERIES_IDS["emae_desest"]: "EMAE"})[["EMAE"]]
    return SeriesResult(dataframe=out, source="datos.gob.ar series API")


def get_ccl_inflation_wages(base_date: str = "2023-12-01") -> SeriesResult:
    ccl, ccl_source = _get_dolar_casa_series("contadoconliqui", "CCL")
    monthly_infl, ipc_source = _get_inflation_monthly()
    infl_index = (1 + (monthly_infl / 100.0).fillna(0)).cumprod().rename("Inflacion")
    wages_df = _get_series_api([SERIES_IDS["salario_registrado"]])
    wages = wages_df[SERIES_IDS["salario_registrado"]].rename("Salarios") if not wages_df.empty else pd.Series(dtype=float, name="Salarios")

    combined = pd.concat([ccl, infl_index, wages], axis=1).dropna(how="all")
    combined = combined.sort_index()
    if combined.empty:
        return SeriesResult(dataframe=combined, source="")

    rebased = pd.DataFrame(index=combined.index)
    for col in ("CCL", "Inflacion", "Salarios"):
        if col in combined:
            rebased[col] = _rebase(combined[col].dropna(), base_date=base_date)

    return SeriesResult(
        dataframe=rebased,
        source=" | ".join(filter(None, [ccl_source, ipc_source, "datos.gob.ar salarios"])),
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
        return SeriesResult(dataframe=combined, source="")

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

    return SeriesResult(
        dataframe=combined,
        source=" | ".join(filter(None, [off_source, ccl_source, "datos.gob.ar ITCRM"])),
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
            return SeriesResult(dataframe=pd.DataFrame(), source="")

        if fiscal.empty or pib.empty:
            return SeriesResult(dataframe=pd.DataFrame(), source="")

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
            return SeriesResult(dataframe=pd.DataFrame(), source="")

        result = pd.DataFrame(index=joined.index)
        result.index.name = "Anio"
        # Both fiscal and PIB are expressed in million ARS at this point.
        result["Resultado_Primario_Pct_PBI"] = (
            joined[SERIES_IDS["fiscal_primario_mensual"]] / joined[SERIES_IDS["pib_anual"]]
        ) * 100
        result["Resultado_Financiero_Pct_PBI"] = (
            joined[SERIES_IDS["fiscal_financiero_mensual"]] / joined[SERIES_IDS["pib_anual"]]
        ) * 100
        return SeriesResult(
            dataframe=result.dropna(how="all"),
            source="datos.gob.ar IMIG + PIB (anual + trimestral)",
        )

    if local_df.empty:
        return SeriesResult(dataframe=pd.DataFrame(), source=source)

    year_col = _find_column(local_df.columns, ("anio", "ano", "year", "fecha"))
    primary_col = _find_column(local_df.columns, ("primario", "resultado_primario", "rdo_primario"))
    fin_col = _find_column(local_df.columns, ("financiero", "resultado_financiero", "rdo_financiero"))
    gdp_col = _find_column(local_df.columns, ("pbi", "gdp"))

    if year_col is None:
        return SeriesResult(dataframe=pd.DataFrame(), source=source)

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

    return SeriesResult(dataframe=result.dropna(how="all"), source=source)


def get_salary_index(base_date: str = "2026-02-01") -> SeriesResult:
    df = _get_series_api([SERIES_IDS["salario_registrado"]])
    if df.empty:
        return SeriesResult(dataframe=pd.DataFrame(), source="")
    out = df.rename(columns={SERIES_IDS["salario_registrado"]: "Indice_Salarios"})[["Indice_Salarios"]]
    infl_m, _ = _get_inflation_monthly()
    if infl_m.empty:
        out["Indice_Salarios_Nominal_Base"] = _rebase(out["Indice_Salarios"], base_date=base_date)
        out["Indice_Salarios_Real_Base"] = out["Indice_Salarios_Nominal_Base"]
        return SeriesResult(dataframe=out, source="datos.gob.ar series API (fallback nominal)")

    infl_idx = (1 + (infl_m / 100.0).fillna(0)).cumprod().rename("IPC")
    infl_aligned = infl_idx.reindex(out.index).ffill()
    out["Indice_Salarios_Real"] = out["Indice_Salarios"] / infl_aligned
    out["Indice_Salarios_Nominal_Base"] = _rebase(out["Indice_Salarios"], base_date=base_date)
    out["Indice_Salarios_Real_Base"] = _rebase(out["Indice_Salarios_Real"], base_date=base_date)
    return SeriesResult(dataframe=out, source="datos.gob.ar series API")


def get_trade_balance_annual() -> SeriesResult:
    result = get_indec_trade_balance()
    df = result.dataframe.copy()
    annual = df.resample("YE").sum(numeric_only=True)
    counts = df["Balanza Comercial"].resample("YE").count()
    annual = annual[counts >= 12]
    annual.index = annual.index.year
    annual.index.name = "Anio"
    return SeriesResult(dataframe=annual, source=result.source_url)


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
        return SeriesResult(dataframe=pd.DataFrame(), source="Yahoo Finance")
    soy = (soy * 0.3674).rename("Soja_USD_ton")
    soy_monthly = soy.resample("MS").mean()

    us_cpi = _us_cpi_monthly(start_date="1990-01-01")
    if us_cpi.empty:
        out = soy_monthly.to_frame()
        out["Soja_Real"] = out["Soja_USD_ton"]
        return SeriesResult(dataframe=out, source="Yahoo Finance (fallback nominal)")

    base_ts = pd.Timestamp(base_date).to_period("M").to_timestamp()
    cpi_base = us_cpi.loc[us_cpi.index <= base_ts]
    if cpi_base.empty:
        cpi_base_value = float(us_cpi.iloc[-1])
    else:
        cpi_base_value = float(cpi_base.iloc[-1])
    cpi_aligned = us_cpi.reindex(soy_monthly.index).ffill()
    real = (soy_monthly * (cpi_base_value / cpi_aligned)).rename("Soja_Real")

    out = pd.concat([soy_monthly, real], axis=1).dropna(how="all")
    return SeriesResult(dataframe=out, source="Yahoo Finance | FRED CPIAUCSL")


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
                return SeriesResult(dataframe=monthly.rename("M2_Palermo").to_frame(), source=str(local_csv))

    raw, source = _pick_first_dataframe(ARGENTINA_DATOS_CANDIDATES["m2_palermo"])
    if raw.empty:
        return SeriesResult(
            dataframe=pd.DataFrame(),
            source=source or "Sin datos. Cargar data/m2_palermo.csv con columnas fecha,m2",
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
        return SeriesResult(dataframe=pd.DataFrame(), source=source)

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if out.empty:
        return SeriesResult(dataframe=pd.DataFrame(), source=source)

    monthly = out.groupby(out[date_col].dt.to_period("M").dt.to_timestamp())[value_col].mean()
    monthly = monthly.rename("M2_Palermo")
    return SeriesResult(dataframe=monthly.to_frame(), source=source)
