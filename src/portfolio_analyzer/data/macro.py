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


def _month_to_number(value: object) -> int | None:
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
    if value is None:
        return None
    normalized = _normalize_text(value)
    if not normalized:
        return None

    for token in normalized.split():
        if token in mapping:
            return mapping[token]
        short = token[:3]
        if short in mapping:
            return mapping[short]

    if re.fullmatch(r"(1[0-2]|0?[1-9])", normalized):
        return int(normalized)
    numeric_match = re.search(r"\bmes\s+(1[0-2]|0?[1-9])\b", normalized)
    if numeric_match:
        return int(numeric_match.group(1))

    return None


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


def _pick_columns(columns: list[str], include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[str]:
    picked: list[str] = []
    for col in columns:
        norm = _normalize_text(col)
        if any(token in norm for token in include) and not any(token in norm for token in exclude):
            picked.append(col)
    return picked


def _column_context_tokens(name: str) -> set[str]:
    tokens = set(_normalize_text(name).split())
    stopwords = {
        "exportaciones",
        "exportacion",
        "importaciones",
        "importacion",
        "periodo",
        "mes",
        "fecha",
        "saldo",
        "balanza",
        "total",
        "mensual",
        "fob",
        "cif",
        "usd",
        "millones",
    }
    return {token for token in tokens if token not in stopwords}


def _column_compatibility_score(
    export_col: str,
    import_col: str,
    first_date_col: str,
    second_date_col: str | None,
) -> int:
    metric_tokens = _column_context_tokens(export_col) & _column_context_tokens(import_col)
    if second_date_col is not None:
        date_tokens = _column_context_tokens(first_date_col) & _column_context_tokens(second_date_col)
    else:
        date_tokens = _column_context_tokens(first_date_col)

    if metric_tokens == date_tokens:
        return 2
    if not metric_tokens and not date_tokens:
        return 2
    if metric_tokens & date_tokens:
        return 1
    return 0


def _is_year_like_series(series: pd.Series) -> bool:
    cleaned = (
        series.ffill()
        .astype(str)
        .str.extract(r"(19\d{2}|20\d{2}|21\d{2})", expand=False)
    )
    ratio = cleaned.notna().mean()
    return bool(ratio >= 0.4)


def _is_month_like_series(series: pd.Series) -> bool:
    parsed = _extract_month(series)
    ratio = parsed.notna().mean()
    return bool(ratio >= 0.4)


def _augment_date_columns_from_content(
    df: pd.DataFrame,
    year_cols: list[str],
    month_cols: list[str],
) -> tuple[list[str], list[str]]:
    years = list(year_cols)
    months = list(month_cols)
    for col in df.columns:
        col_name = str(col)
        if col_name not in years and _is_year_like_series(df[col]):
            years.append(col_name)
        if col_name not in months and _is_month_like_series(df[col]):
            months.append(col_name)
    return years, months


def _extract_month(series: pd.Series) -> pd.Series:
    month_from_name = series.astype(str).apply(_month_to_number)
    month_from_num = pd.to_numeric(
        series.astype(str).str.extract(r"^\s*(1[0-2]|0?[1-9])\s*$", expand=False),
        errors="coerce",
    )
    month = month_from_name.where(month_from_name.notna(), month_from_num)
    return pd.to_numeric(month, errors="coerce")


def _parse_indec_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(raw_df)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        return pd.DataFrame()

    columns = list(df.columns)
    export_cols = _pick_columns(
        columns,
        include=("export",),
        exclude=("acumul", "variac", "indice", "precio", "interan"),
    )
    import_cols = _pick_columns(
        columns,
        include=("import",),
        exclude=("acumul", "variac", "indice", "precio", "interan"),
    )
    balance_cols = _pick_columns(columns, include=("saldo", "balanza"))
    year_cols = _pick_columns(columns, include=("periodo", "ano", "anio", "año"), exclude=("mes",))
    month_cols = _pick_columns(columns, include=("mes",))
    date_cols = _pick_columns(columns, include=("fecha",))
    year_cols, month_cols = _augment_date_columns_from_content(df, year_cols, month_cols)

    if not export_cols or not import_cols:
        return pd.DataFrame()

    best_candidate = pd.DataFrame()
    best_score: tuple[pd.Timestamp, int, int] = (pd.Timestamp.min, -1, 0)
    date_pairs: list[tuple[str, str | None]] = []
    for year_col in year_cols:
        for month_col in month_cols:
            date_pairs.append((year_col, month_col))
    for date_col in date_cols:
        date_pairs.append((date_col, None))
    for year_col in year_cols:
        date_pairs.append((year_col, None))

    if not date_pairs:
        return pd.DataFrame()

    for export_col in export_cols:
        for import_col in import_cols:
            if export_col == import_col:
                continue
            export_tokens = _column_context_tokens(export_col)
            import_tokens = _column_context_tokens(import_col)
            if export_tokens != import_tokens and (export_tokens or import_tokens):
                continue
            base = pd.DataFrame()
            base["Exportaciones"] = _to_numeric_series(df[export_col])
            base["Importaciones"] = _to_numeric_series(df[import_col])

            balance_col = None
            for candidate in balance_cols:
                norm_candidate = _normalize_text(candidate)
                # Prefer monthly balance-like columns when available.
                if "mensual" in norm_candidate:
                    balance_col = candidate
                    break
            if balance_col is None and balance_cols:
                balance_col = balance_cols[0]

            if balance_col:
                base["Balanza Comercial"] = _to_numeric_series(df[balance_col])
            else:
                base["Balanza Comercial"] = base["Exportaciones"] - base["Importaciones"]

            for first_col, second_col in date_pairs:
                parsed = base.copy()
                if second_col is not None:
                    years = (
                        df[first_col]
                        .ffill()
                        .astype(str)
                        .str.extract(r"(\d{4})", expand=False)
                        .astype(float)
                    )
                    months = _extract_month(df[second_col])
                    parsed["Fecha"] = pd.to_datetime(
                        years.fillna(0).astype(int).astype(str)
                        + "-"
                        + months.fillna(0).astype(int).astype(str)
                        + "-01",
                        errors="coerce",
                    )
                elif "fecha" in _normalize_text(first_col):
                    parsed["Fecha"] = pd.to_datetime(df[first_col], errors="coerce", dayfirst=True)
                else:
                    parsed["Fecha"] = pd.to_datetime(df[first_col], errors="coerce", dayfirst=True)

                parsed = parsed.dropna(subset=["Fecha", "Exportaciones", "Importaciones"]).copy()
                parsed = parsed[
                    (parsed["Fecha"].dt.year >= 1990)
                    & (parsed["Fecha"].dt.year <= pd.Timestamp.now().year + 1)
                ].copy()
                if parsed.empty:
                    continue

                parsed = parsed.drop_duplicates(subset=["Fecha"], keep="last")
                parsed = parsed.sort_values("Fecha").set_index("Fecha")
                parsed = parsed[["Exportaciones", "Importaciones", "Balanza Comercial"]]
                compatibility = _column_compatibility_score(export_col, import_col, first_col, second_col)
                score = (parsed.index.max(), compatibility, len(parsed))
                if score > best_score:
                    best_candidate = parsed
                    best_score = score

    return best_candidate


def _score_indec_candidate(df: pd.DataFrame) -> tuple[pd.Timestamp, int]:
    if df.empty:
        return (pd.Timestamp.min, 0)
    return (df.index.max(), len(df))


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
    best_candidate: pd.DataFrame | None = None
    best_score: tuple[pd.Timestamp, int] = (pd.Timestamp.min, 0)

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
                    score = _score_indec_candidate(parsed)
                    if score > best_score:
                        best_candidate = parsed
                        best_score = score
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

    if best_candidate is not None:
        return INDECTradeResult(dataframe=best_candidate, source_url=INDEC_EXCEL_URL)

    raise RuntimeError(
        "INDEC file format changed and could not be parsed with known layouts. "
        f"Last parser error: {last_error}"
    )
