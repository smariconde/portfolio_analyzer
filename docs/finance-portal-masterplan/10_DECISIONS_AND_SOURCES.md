# Decisiones, riesgos y fuentes

Investigacion revisada el 20 de agosto de 2026. Los enlaces y condiciones comerciales deben revalidarse al iniciar el repo y antes del lanzamiento.

## Decisiones tomadas

### ADR-001 resumido

| Tema | Decision | Motivo | Revisar cuando |
|---|---|---|---|
| Arquitectura | monolito modular Next.js | deploy simple y limites claros | jobs excedan Functions |
| Runtime | TypeScript/Node en Vercel | una sola toolchain; dominio portable | calculo requiera stack cientifico no viable |
| DB | Postgres serverless + Drizzle | series/lineage/relaciones y SQL | escala analitica lo justifique |
| UI | shadcn + Tailwind | componentes editables y accesibles | no aplica |
| Charts | Recharts/shadcn primero | menor complejidad | scatter real no rinda |
| Table | TanStack Table | filtros/columnas controlables | no aplica |
| IA | Vercel AI SDK + OpenRouter | provider abstraction, streaming y modelos multiples | SLA/costo aconseje directo |
| Research | Tavily acotado | search/extract con dominios | fuentes directas cubran todo |
| Market data | adapter, FMP como primer spike | statements + historical market cap | cobertura/licencia/costo fallen |
| Auth/BYOK | posterior | evita complejidad y riesgo prematuro | antes de usuarios externos |
| Valuacion | formulas puras + IA para supuestos | reproducibilidad y control | nunca invertir esta regla |

## Conocimiento rescatado del repo actual

El repo `portfolio_analyzer` confirma que estas ideas son viables y ya tiene pruebas conceptuales:

- BCRA API v4 con series, fechas y ajustes;
- balanza comercial INDEC con parser tolerante a layouts;
- dashboard argentino con EMAE, salarios, ITCRM, fiscal, inflacion/REM y liquidez;
- spread soja con Rosario, FX BNA, conversion y control de outliers;
- scanner sectorial/CEDEAR a 2/5 anos;
- valuacion V2 deterministica con FCFF, DDM, residual income, normalizacion ciclica, escenarios y Monte Carlo.

No copiar automaticamente esos modulos Python. Usarlos como inventario de requisitos y fixtures conceptuales; redisenar contratos, fuentes y formulas en TypeScript con la metodologia de este plan.

## Fuentes de plataforma

- Next.js, Backend for Frontend: https://nextjs.org/docs/app/guides/backend-for-frontend
- Next.js, caching/revalidation: https://nextjs.org/docs/app/getting-started/revalidating
- Vercel Functions: https://vercel.com/docs/functions
- Vercel max duration: https://vercel.com/docs/functions/configuring-functions/duration
- Vercel Cron: https://vercel.com/docs/cron-jobs y https://vercel.com/docs/cron-jobs/manage-cron-jobs
- Vercel Storage/Marketplace: https://vercel.com/docs/storage y https://vercel.com/docs/marketplace-storage
- Vercel environment variables: https://vercel.com/docs/environment-variables
- Sensitive environment variables: https://vercel.com/docs/environment-variables/sensitive-environment-variables

Hechos relevantes: Functions/Route Handlers no deben asumir filesystem persistente ni memoria compartida; cron llama un endpoint HTTP, usa UTC, no reintenta una falla y comparte limites de Function; el Postgres nuevo se provisiona mediante Marketplace.

## IA e investigacion

- OpenRouter + Vercel AI SDK: https://openrouter.ai/docs/guides/community/vercel-ai-sdk
- Structured outputs: https://openrouter.ai/docs/guides/features/structured-outputs
- Provider routing: https://openrouter.ai/docs/guides/routing/provider-selection
- Zero Data Retention: https://openrouter.ai/docs/guides/features/zdr
- Guardrails/budgets: https://openrouter.ai/docs/guides/features/guardrails/overview
- Tavily JS: https://docs.tavily.com/sdk/javascript/quick-start
- Tavily Search: https://docs.tavily.com/documentation/api-reference/endpoint/search

Structured output depende del modelo/provider; usar `require_parameters`, schema estricto y fallback controlado. ZDR limita endpoints disponibles. Tavily permite filtros de dominio y fechas, pero la app debe conservar las fuentes primarias y no tratar su answer sintetica como dato financiero.

## Datos corporativos y mercado

- SEC EDGAR APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- FMP docs: https://site.financialmodelingprep.com/developer/docs
- Twelve Data fundamentals: https://twelvedata.com/fundamentals
- Financial Datasets: https://www.financialdatasets.ai/ y https://www.financialdatasets.ai/pricing

SEC ofrece submissions y XBRL/companyfacts sin API key y bulk nightly, pero no soporta CORS directo y exige Fair Access. FMP documenta market cap historica, estados, precios y growth endpoints. Twelve Data declara market cap historica y cobertura global/XBUE. Financial Datasets enfatiza datos trazados a filings, pero su costo/cobertura/licencia deben compararse en el spike.

## CEDEAR y Argentina

- Caja de Valores, listado CEDEAR/ratios: https://cajadevalores.com.ar/Servicios/Cedears
- Banco Comafi, programas: https://www.comafi.com.ar/Programas-CEDEARs-2483.note.aspx
- BYMA CEDEAR: https://www.byma.com.ar/productos/productos-financieros/cedears
- BCRA APIs: https://www.bcra.gob.ar/apis-banco-central/
- Manual BCRA monetarias v4: https://www.bcra.gob.ar/Catalogo/Content/files/pdf/principales-variables-v4.pdf
- API Series de Tiempo Argentina: https://www.argentina.gob.ar/datos-abiertos/api-series-de-tiempo
- Camara Arbitral/BCR pizarra: https://www.cac.bcr.com.ar/es/precios-de-pizarra
- BCR mercado disponible: https://www.bcr.com.ar/es/mercados/mercado-de-granos/cotizaciones/cotizaciones-locales/mercado-fisico-de-rosario
- MAGyP historicos de precios camara: https://www.magyp.gob.ar/sitio/areas/ss_mercados_agropecuarios/areas/granos/

BCRA v4 es la version vigente a usar; versiones previas fueron deprecadas. El manual fija paginacion y limite maximo. CEDEAR ratios deben historizarse porque cambian. La pizarra Rosario se publica en ARS/t y su conversion informativa usa BNA comprador.

## Damodaran y valuacion

- Portal/clase de valuacion: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/equity.html
- Materiales y model chooser: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valuation/val.htm
- Spreadsheets por metodo: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/eqspread.htm
- Numbers and Narrative: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/numbers%26narrative.htm
- Normalizing earnings: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/normearn.htm
- Commodity value drivers: https://pages.stern.nyu.edu/adamodar/New_Home_Page/littlebook/commodityvaluedrivers.htm
- Financial services value drivers: https://pages.stern.nyu.edu/adamodar/New_Home_Page/littlebook/bankvaluedriver.htm
- Current datasets: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html
- Country risk premiums: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html
- Historical implied ERP: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html

Principios incorporados: metodo segun empresa/lifecycle; normalizacion a lo largo de un ciclo; excess return para financieras; conexion narrativa-numeros; riesgo, reinversion y terminal coherentes; ERP/CRP con fecha.

## Skills y operacion del agente

- skills.sh docs: https://www.skills.sh/docs
- skills CLI: https://www.skills.sh/docs/cli
- Vercel Next skills: https://github.com/vercel-labs/next-skills
- Vercel React best practices: https://www.skills.sh/vercel-labs/agent-skills/react-best-practices
- Vercel AI SDK skill: https://github.com/vercel/ai/blob/main/skills/use-ai-sdk/SKILL.md

El CLI soporta instalacion por skill y lockfile. La documentacion de skills.sh dice expresamente que no garantiza todos los paquetes: revisar antes de instalar y preferir fuentes oficiales/versionadas.

## Riesgos abiertos

1. Licencia de redistribucion del proveedor de market data.
2. Feed estable/licenciado de Chicago y politica de contrato continuo.
3. Cobertura de foreign filers/ADRs y reconciliacion IFRS/US-GAAP.
4. MEP/CCL con fuente contractual, no agregador fragil.
5. Precision/frescura real de CEDEAR downloads y cambios de ratio.
6. Modelo de auth/vault para BYOK.
7. Region optima DB/Functions medida con usuarios/proveedores reales.
8. Limites/costos del plan Vercel y cron elegido.

Estos puntos tienen ADR/spike antes de convertirse en dependencia productiva.

