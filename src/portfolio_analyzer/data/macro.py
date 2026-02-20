from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bcra import (
    VARIABLES_BCRA,
    adjust_data_by_index,
    fetch_bcra_data,
    get_cer_data,
    get_minorista_exchange_rate,
)

INDEC_EXCEL_URL = "https://www.indec.gob.ar/ftp/cuadros/economia/balanmensual.xls"


@dataclass
class BCRAResult:
    variable_id: int
    variable_name: str
    dataframe: pd.DataFrame


@dataclass
class INDECTradeResult:
    dataframe: pd.DataFrame
    source_url: str


def bcra_variables() -> dict[int, str]:
    return VARIABLES_BCRA.copy()


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["fecha", "valor"]).set_index("fecha")

    df = pd.DataFrame(records).copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["fecha", "valor"]).sort_values("fecha")
    return df.set_index("fecha")


def get_bcra_series(
    variable_id: int,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 3000,
    adjustment: str = "none",
) -> BCRAResult:
    if variable_id not in VARIABLES_BCRA:
        raise ValueError(f"Unknown BCRA variable id: {variable_id}")

    records = fetch_bcra_data(
        variable_id=variable_id,
        fecha_desde=start_date,
        fecha_hasta=end_date,
        limit=limit,
    )
    if not records:
        raise RuntimeError("No data returned from BCRA API.")

    if adjustment not in {"none", "usd", "cer"}:
        raise ValueError("adjustment must be one of: none, usd, cer")

    if adjustment == "usd":
        index_data = get_minorista_exchange_rate(start_date, end_date)
        records = adjust_data_by_index(records, index_data, index_name="Tipo de Cambio Minorista")
    elif adjustment == "cer":
        index_data = get_cer_data(start_date, end_date)
        records = adjust_data_by_index(records, index_data, index_name="CER")

    df = _records_to_df(records)
    if df.empty:
        raise RuntimeError("Series is empty after processing.")

    return BCRAResult(
        variable_id=variable_id,
        variable_name=VARIABLES_BCRA[variable_id],
        dataframe=df,
    )


def _month_to_number(value: str) -> int | None:
    mapping = {
        "enero": 1,
        "febrero": 2,
        "marzo": 3,
        "abril": 4,
        "mayo": 5,
        "junio": 6,
        "julio": 7,
        "agosto": 8,
        "septiembre": 9,
        "octubre": 10,
        "noviembre": 11,
        "diciembre": 12,
        "ene": 1,
        "feb": 2,
        "mar": 3,
        "abr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "ago": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dic": 12,
    }
    if not isinstance(value, str):
        return None
    return mapping.get(value.strip().lower())


def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flat_cols: list[str] = []
    for col in out.columns:
        if isinstance(col, tuple):
            parts = [str(p).strip() for p in col if str(p).strip() and "unnamed" not in str(p).lower()]
            flat_cols.append(" ".join(parts) if parts else str(col))
        else:
            flat_cols.append(str(col).strip())
    out.columns = flat_cols
    return out


def _to_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(r"[^\d,.\-]", "", regex=True)
    )
    both = cleaned.str.contains(",", regex=False) & cleaned.str.contains(".", regex=False)
    cleaned.loc[both] = cleaned.loc[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    cleaned.loc[~both] = cleaned.loc[~both].str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def _pick_column(columns: list[str], include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> str | None:
    for col in columns:
        norm = _normalize_text(col)
        if any(token in norm for token in include) and not any(token in norm for token in exclude):
            return col
    return None


def _extract_month(series: pd.Series) -> pd.Series:
    month_from_name = series.astype(str).apply(_month_to_number)
    month_from_num = pd.to_numeric(series, errors="coerce")
    month = month_from_name.where(month_from_name.notna(), month_from_num)
    return pd.to_numeric(month, errors="coerce")


def _parse_indec_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(raw_df)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        return pd.DataFrame()

    columns = list(df.columns)
    export_col = _pick_column(columns, include=("export",), exclude=("acumul",))
    import_col = _pick_column(columns, include=("import",), exclude=("acumul",))
    balance_col = _pick_column(columns, include=("saldo", "balanza"))
    year_col = _pick_column(columns, include=("periodo", "ano", "anio", "año"), exclude=("mes",))
    month_col = _pick_column(columns, include=("mes",))
    date_col = _pick_column(columns, include=("fecha",))

    if not export_col or not import_col:
        return pd.DataFrame()

    parsed = pd.DataFrame()
    parsed["Exportaciones"] = _to_numeric_series(df[export_col])
    parsed["Importaciones"] = _to_numeric_series(df[import_col])

    if balance_col:
        parsed["Balanza Comercial"] = _to_numeric_series(df[balance_col])
    else:
        parsed["Balanza Comercial"] = parsed["Exportaciones"] - parsed["Importaciones"]

    if date_col:
        parsed["Fecha"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    elif year_col and month_col:
        years = (
            df[year_col]
            .ffill()
            .astype(str)
            .str.extract(r"(\d{4})", expand=False)
            .astype(float)
        )
        months = _extract_month(df[month_col])
        parsed["Fecha"] = pd.to_datetime(
            years.fillna(0).astype(int).astype(str)
            + "-"
            + months.fillna(0).astype(int).astype(str)
            + "-01",
            errors="coerce",
        )
    elif year_col:
        # Sometimes period already contains month + year in one field.
        parsed["Fecha"] = pd.to_datetime(df[year_col], errors="coerce", dayfirst=True)
    else:
        return pd.DataFrame()

    parsed = parsed.dropna(subset=["Fecha", "Exportaciones", "Importaciones"]).copy()
    if parsed.empty:
        return parsed

    parsed = parsed.sort_values("Fecha").set_index("Fecha")
    parsed = parsed[["Exportaciones", "Importaciones", "Balanza Comercial"]]
    return parsed


def get_indec_trade_balance() -> INDECTradeResult:
    response = requests.get(INDEC_EXCEL_URL, timeout=30)
    response.raise_for_status()

    excel_content = BytesIO(response.content)
    try:
        # Smoke check that xlrd is installed.
        pd.read_excel(BytesIO(response.content), sheet_name=0, nrows=1, engine="xlrd")
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'xlrd' (>=2.0.1) required to parse INDEC .xls file. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    # Try several header configurations because INDEC periodically changes the layout.
    header_candidates: list[object] = [[2, 3], [1, 2], [3, 4], 2, 1, 0, None]
    sheet_candidates: list[object] = ["FOB-CIF", 0]
    last_error: Exception | None = None

    for sheet in sheet_candidates:
        for header in header_candidates:
            try:
                df_raw = pd.read_excel(
                    BytesIO(response.content),
                    sheet_name=sheet,
                    header=header,
                    engine="xlrd",
                )
                parsed = _parse_indec_dataframe(df_raw)
                if not parsed.empty:
                    return INDECTradeResult(dataframe=parsed, source_url=INDEC_EXCEL_URL)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

    raise RuntimeError(
        "INDEC file format changed and could not be parsed with known layouts. "
        f"Last parser error: {last_error}"
    )
