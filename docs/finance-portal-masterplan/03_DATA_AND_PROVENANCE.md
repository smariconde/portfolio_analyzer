# Estrategia de datos y trazabilidad

## Regla de oro

Un dato financiero no es solo un numero. Su identidad incluye concepto, entidad, periodo, fecha de presentacion, unidad, moneda, fuente, transformacion y fecha de descarga. Si alguno falta, el sistema baja su quality score o lo rechaza.

## Jerarquia de fuentes

| Dominio | Fuente primaria | Adaptador inicial | Fallback/validacion |
|---|---|---|---|
| Filings EE.UU. | SEC EDGAR | SEC submissions/companyfacts | filing HTML/XBRL |
| Fundamentales normalizados | proveedor licenciado | FMP sugerido | SEC + segundo proveedor |
| Precios/market cap historica | proveedor licenciado | FMP sugerido | Twelve Data/Financial Datasets |
| CEDEAR | Caja de Valores/Comafi | ingesta versionada | BYMA/anuncios |
| BCRA | API oficial v4 | REST | catalogo/metodologia BCRA |
| Macro Argentina | datos.gob.ar/INDEC | API/descarga oficial | release original |
| FX referencia | BCRA/BNA | API/descarga | ninguna silenciosa |
| Soja Rosario | BCR/Camara Arbitral | descarga/parser | MAGyP historicos |
| Soja Chicago | feed con licencia | adapter configurable | ninguno no autorizado |
| Riesgo/sectores | NYU Stern/Damodaran | snapshot parser | FRED/Treasury para Rf |
| Evidencia cualitativa | documentos primarios | Tavily search/extract | filings/IR directos |

El proveedor sugerido puede cambiar luego de una prueba de cobertura, costo, rate limits, historico y licencia. La UI y el dominio no cambian.

## Registro de fuentes

Cada fuente necesita una entrada versionada con:

- owner y URL de documentacion;
- dataset/endpoints utilizados;
- autenticacion y rate limits;
- terminos, licencia y permiso de cache/redistribucion;
- frecuencia esperada y freshness SLA;
- zona horaria, moneda y unidades;
- parser version y fixtures;
- fallback permitido;
- contacto/status page;
- fecha de ultima revision legal/tecnica.

No marcar `redistributable=true` por inferencia. Hasta verificar terminos, exponer calculos derivados y metadatos minimos, no payloads completos.

## SEC

- Usar CIK como identidad, no ticker.
- Enviar User-Agent identificable y respetar Fair Access.
- Guardar `accession`, `filed`, `form`, `fy`, `fp`, `start`, `end`, `frame`, `unit` y taxonomy tag.
- Resolver tags alternativos con un mapping canonico versionado; conservar el tag original.
- Para puntos historicos evitar duplicados de filings posteriores y amendments sin perder lineage.
- Los bulk archives son preferibles para universo grande; las APIs por empresa sirven para refresh/detalle.
- SEC no entrega precios ni market cap historica: no forzar ese uso.

## Proveedor normalizado

El spike de proveedor debe medir al menos 30 tickers representativos: mega-cap, ADR/foreign filer, banco, aseguradora, REIT, commodity, empresa con perdidas, split reciente y 10 CEDEARs. Comparar 20 campos con filing fuente y evaluar:

- cobertura anual/trimestral/TTM de 5+ anos;
- EPS diluido, shares diluidas, market cap historica y corporate actions;
- moneda de reporte y conversion;
- sector/industria e identificadores;
- revisiones/restatements;
- batching, cuota, latencia y costo;
- derechos de uso en web publica.

No construir el producto sobre datos gratis de Yahoo obtenidos por endpoints no contractuales.

## Registro CEDEAR

Modelo minimo:

- `cedear_symbol`, `underlying_symbol`, `underlying_isin`, `cedear_isin`;
- `program_name`, `issuer`, `origin_exchange`, `country`;
- ratio como fraccion exacta `cedear_units / underlying_units`, no string solamente;
- alcance de inversor, tipo (share/ETF/corporate), sector publicado;
- `valid_from`, `valid_to`, fuente, hash del documento y fecha de ingesta.

Los ratios cambian. Nunca sobrescribir historia. La etiqueta CEDEAR del screener depende del snapshot vigente a la fecha consultada.

## Screener y formulas

Crear un metric catalog con nombre, formula, periodicidad, tratamiento de negativos y version. Ejemplos:

- `roic = nopat / average_invested_capital`;
- `fcf_yield = fcff_or_equity_fcf / corresponding_market_value` con definicion explicita;
- `net_debt_to_ebitda = (debt - excess_cash) / ebitda`;
- `share_count_cagr` sobre acciones diluidas promedio ajustadas.

No comparar ratios sin significado entre sectores. Por ejemplo, deuda/EBITDA no es el filtro central de un banco; mostrar metricas sectoriales o `not_applicable`.

## Fundamental gap: especificacion exacta

Para horizonte `h`:

1. Seleccionar el ultimo cierre fiscal disponible y el cierre fiscal comparable mas cercano a `h` anos atras.
2. Exigir que la distancia real este dentro de una tolerancia documentada; usar `years = days / 365.2425`.
3. Market cap debe corresponder a la fecha fiscal o al ultimo dia de mercado cercano, con ventana maxima.
4. EPS es diluted EPS de operaciones continuas cuando pueda normalizarse; mostrar GAAP reportado como control.
5. Si ambos extremos son positivos:
   - `market_cap_cagr_pct = ((mc1 / mc0)^(1/years)-1)*100`
   - `eps_cagr_pct = ((eps1 / eps0)^(1/years)-1)*100`
   - `fundamental_gap_pp = eps_cagr_pct - market_cap_cagr_pct`
6. EPS no positivo produce categoria especial, no CAGR artificial.
7. Calcular tambien `net_income_cagr_pct` y `diluted_shares_cagr_pct`.

Interpretacion: gap positivo indica expansion de beneficio por accion superior a market cap en ese intervalo. No prueba infravaloracion: puede reflejar pico ciclico, riesgo, deterioro esperado, one-offs o cambio de mix.

## Argentina

### Series iniciales

- BCRA: reservas, base monetaria, M2, depositos, prestamos, tasas, CER/UVA, inflacion publicada y REM cuando esten en v4.
- Datos/INDEC: EMAE, salarios, IPC, PIB, industria/construccion, exportaciones, importaciones y saldo comercial.
- Cambiario: oficial mayorista/minorista de fuente oficial; MEP/CCL solo con feed licenciado y metodologia visible.
- Fiscal: resultado primario y financiero, montos y ratios sobre PIB con denominadores compatibles.
- Competitividad: ITCRM y series rebased sin presentar el indice rebaseado como nivel oficial.

### Soja

- Rosario: precio pizarra ARS/t y, si se calcula USD/t, usar FX BNA comprador de la misma fecha o mostrar el descalce.
- Chicago: precio y contrato/fuente identificados. Conversion, si parte de USD/bushel: `USD_per_metric_ton = USD_per_bushel * 36.7437`.
- Definir `basis_pct = (rosario_usd_t / chicago_usd_t - 1) * 100`; negativo es descuento Rosario.
- Si se usa futuro continuo, guardar contrato y politica de roll. Los saltos de roll no son movimiento economico spot.
- Mostrar correlacion, percentil, mediana, bandas robustas y outliers; no convertir percentil en orden de compra/venta.

## Calidad y fallbacks

Flags sugeridos:

- `stale`, `estimated`, `restated`, `currency_converted`, `unit_converted`;
- `fiscal_alignment_approximate`, `missing_period`, `provider_disagreement`;
- `non_comparable`, `negative_denominator`, `outlier`, `parser_schema_changed`;
- `license_restricted`, `fallback_source`.

Un fallback se muestra como fallback. Un dato faltante es `null` con razon; nunca cero.

