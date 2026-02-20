# Portfolio Analyzer

Toolkit de análisis financiero con una interfaz principal en Streamlit y comandos CLI para automatización.

## Qué incluye

- `Portfolio Agent`: análisis multiagente por ticker con salida estructurada.
- `Portfolio Optimizer`: optimización de asignación (Sortino-oriented).
- `Valuation Agent`: valuación DCF asistida por LLM.
- `SEC Agent`: resumen de filings 10-K/10-Q + PDF.
- `Sortino Scanner`: ranking global + shortlist CEDEAR + vista visual con charts.
- `Commodities Spread`: seguimiento Chicago vs Rosario.
- `BCRA Macro` e `INDEC Trade Balance`: paneles macro Argentina.

## Estructura del repo

```text
.
├── ai_agent/                  # App Streamlit (Home + pages)
├── src/portfolio_analyzer/    # Núcleo reusable (analytics, data, cli, core)
├── scripts/                   # Scripts históricos y wrappers
├── outputs/                   # Salidas generadas (sortino, commodities, etc.)
├── cache/                     # Cache de datos de mercado
├── tests/                     # Tests
├── pyproject.toml
├── requirements.txt
└── Makefile
```

## Instalación rápida

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Variables de entorno (`.env`)

```env
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
FINANCIAL_DATASETS_API_KEY=...

# Opcional: override de modelo Gemini
GOOGLE_GENAI_MODEL=gemini-2.5-flash
# o lista de fallback
GOOGLE_GENAI_MODELS=gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash
```

Notas:
- Si solo tenés `GEMINI_API_KEY`, la app la toma como fallback para `GOOGLE_API_KEY`.
- `xlrd>=2.0.1` es requerido para leer el `.xls` de INDEC.

## Ejecutar la app

```bash
source .venv/bin/activate
streamlit run ai_agent/Home.py
```

También:

```bash
make ui
portfolio-analyzer ui
```

## Comandos útiles

```bash
make setup       # instala deps + editable + herramientas de dev
make test        # pytest -q
make lint        # ruff check src tests
make check       # lint + test
make sortino
make optimizer
make commodities
```

## CLI (ejemplos)

```bash
portfolio-analyzer run sortino
portfolio-analyzer run optimizer --tickers AAPL,MSFT,NVDA --portfolio-size 40000
portfolio-analyzer run commodities --start-date 2024-01-01
portfolio-analyzer run agent --ticker AAPL --start-date 2025-01-01 --end-date 2025-12-31
```

## Troubleshooting rápido

- `404 NOT_FOUND model ...`: configurá `GOOGLE_GENAI_MODEL` o `GOOGLE_GENAI_MODELS`.
- `Missing optional dependency 'xlrd'`: `pip install -r requirements.txt`.
- `INDEC file format changed`: el parser ya es flexible; si reaparece, compartir el error completo.
- Si una página no refleja cambios recientes: reiniciar Streamlit.
