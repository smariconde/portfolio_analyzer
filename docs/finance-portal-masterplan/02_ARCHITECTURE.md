# Arquitectura propuesta

## Decision principal

Un monolito modular Next.js en una sola region primaria, desplegado en Vercel, con PostgreSQL serverless cercano a las Functions. Es la opcion de menor complejidad que conserva limites claros y permite extraer workers o servicios mas adelante.

```text
Browser
  -> Next.js App Router
       -> Server Components (lecturas y composicion)
       -> Route Handlers (API, cron, IA streaming)
       -> Server Actions (mutaciones UI)
          -> Application services
             -> Pure domain (ratios, growth, valuation, policies)
             -> Provider ports
                -> SEC / market data / BCRA / INDEC / CEDEAR / Tavily / OpenRouter
             -> Repositories
                -> Postgres
```

## Estructura sugerida

```text
src/
  app/
    (portal)/
    api/
  components/
    ui/
    charts/
  modules/
    companies/
      domain/
      application/
      infrastructure/
      ui/
    screeners/
    growth-gap/
    valuation/
    argentina/
    research/
  server/
    db/
    providers/
    ai/
    observability/
    security/
  shared/
    domain/
    contracts/
    utils/
drizzle/
tests/
  fixtures/
  contract/
  e2e/
docs/
.agents/skills/
```

`domain/` no importa React, Next, Drizzle ni SDKs externos. `infrastructure/` implementa ports. `ui/` consume DTOs de aplicacion, no respuestas crudas.

## Contratos centrales

```ts
type Provenance = {
  sourceId: string;
  sourceUrl?: string;
  asOf: string;
  fetchedAt: string;
  unit: string;
  currency?: string;
  period?: "instant" | "quarter" | "annual" | "ttm" | "daily" | "monthly";
  qualityFlags: string[];
  ingestionRunId: string;
};

type Observation<T = string> = {
  value: T;
  rawValue?: string;
  provenance: Provenance;
};

interface FundamentalsProvider {
  searchCompanies(query: string): Promise<CompanyRef[]>;
  getStatements(symbol: string, options: PeriodOptions): Promise<StatementSet>;
  getMetrics(symbol: string, options: PeriodOptions): Promise<MetricSeries[]>;
}

interface MarketDataProvider {
  getPriceHistory(symbol: string, range: DateRange): Promise<PricePoint[]>;
  getMarketCapHistory(symbol: string, range: DateRange): Promise<MarketCapPoint[]>;
}

interface MacroSeriesProvider {
  listSeries(): Promise<SeriesDefinition[]>;
  getSeries(id: string, range: DateRange): Promise<MacroObservation[]>;
}
```

Los schemas Zod reales son la fuente de verdad de runtime. Los tipos anteriores expresan la intencion.

## Limite web/backend

- Server Components llaman application services directamente.
- Route Handlers existen para clientes externos, interaccion client-side, cron o streaming.
- Server Actions mutan watchlists, configuracion o supuestos; no se usan para lecturas generales porque se serializan/encolan.
- Toda importacion de DB, API key o proveedor comienza con `import "server-only"`.
- DTOs de salida minimizan campos y remueven secretos, prompts internos y payloads licenciados no redistribuibles.

## Persistencia minima

- `source_registry`: fuente, owner, URL, licencia, redistribucion, auth, SLA esperado.
- `ingestion_runs`: fuente, version de parser, estado, cursor, counts, errores, timestamps.
- `instruments`: identidad estable, ticker, exchange, CIK/ISIN, sector/industria.
- `cedear_programs`: simbolo BYMA, subyacente, ratio, emisor, vigencia y snapshot.
- `company_snapshots`: perfil y clasificacion con `valid_from/valid_to`.
- `financial_facts`: metrica canonica, periodo, valor, unidad, moneda y provenance.
- `market_cap_points` y `price_points`: series normalizadas.
- `macro_series` y `macro_observations`.
- `source_documents`: URL, hash, fecha de publicacion/extraccion y metadata.
- `valuation_runs`: input snapshot, engine version, method, status y result JSON.
- `valuation_assumptions`: valor, origen, evidencia, override y validaciones.
- `research_evidence`: fragmento pequeno, titulo, URL, fecha y hash.
- `users`, `saved_views`, `provider_credentials`: solo en la fase de cuentas/BYOK.

Usar claves naturales solo para lookup. Las relaciones internas usan IDs estables porque tickers, ratios CEDEAR y nombres cambian.

## Ingesta y caching

- Request path: leer cache/DB y hacer refresh acotado solo cuando sea seguro.
- Cron: descargar, validar, guardar staging, comparar y publicar atomicamente.
- Cada job es idempotente por `(source, dataset, as_of, parser_version)`.
- Guardar ultimo snapshot valido. Un parser roto no reemplaza datos buenos por vacio.
- Vercel Cron no reintenta automaticamente: persistir fallas y reintentar en la siguiente corrida o adoptar una cola durable cuando el volumen lo justifique.
- TTL orientativo: quotes 1-15 min; market cap/fundamentals 6-24 h; BCRA/BNA/BCR 6-24 h; CEDEAR diario; Damodaran semanal con cambio efectivo anual/trimestral.

## Escalabilidad sin sobreingenieria

Extraer un worker cuando una ingesta no quepa de forma confiable en el limite de Function, necesite fan-out/retries fuertes o consuma CPU sostenida. Opciones futuras: QStash, Inngest, DBOS o Vercel Workflow, evaluadas con una ADR. No introducir Kafka, Kubernetes ni una segunda base para el MVP.

## Regiones

Elegir la region de Functions junto con la region del Postgres. Los proveedores estadounidenses pueden favorecer `iad1`, mientras que usuarios argentinos favorecen menor latencia al sur; medir antes de decidir. El cache/CDN reduce el impacto para lecturas publicas.

