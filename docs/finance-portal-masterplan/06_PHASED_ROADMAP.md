# Roadmap por fases

Cada fase termina desplegable y con un gate. Las duraciones son orientativas; importa el orden de dependencias.

## Fase 0 - Fundacion y decisiones

Entregables:

- repo Next.js estricto, package manager y CI;
- PRD, arquitectura, ADR-001, source registry y metodologia;
- `AGENTS.md`, skills auditadas y scripts estandarizados;
- `.env.example`, env validation y threat/data-license checklist;
- wireframes simples y design tokens;
- backlog de fases.

Gate: `pnpm lint`, `typecheck`, `test` y build pasan; no hay secrets; decisiones costosas quedan explicitas.

## Fase 1 - Vertical slice sin proveedor real

Entregables:

- shell, navegacion, setup/health y estados de error;
- Postgres/Drizzle y tablas base;
- provider ports + fake deterministico;
- una compania fixture y FCFF base con sensibilidad;
- source/freshness UI;
- unit, contract y E2E del flujo demo.

Gate: deploy preview reproducible; misma fixture produce mismo hash/resultado.

## Fase 2 - Empresas, CEDEAR y screener

Entregables:

- spike y ADR de proveedor; integracion aprobada;
- SEC identity/audit adapter;
- ingesta versionada de CEDEAR;
- metric catalog y screener 2Y/5Y;
- CSV con metadata; provider degradation.

Gate: muestra de 30 empresas reconciliada; 100% de filas con as-of/source; licencia revisada.

## Fase 3 - Fundamental gap

Entregables:

- pipeline fiscal-aligned de market cap/EPS;
- categorias de EPS no comparable;
- scatter, tabla, filtros y detail drawer;
- controles net income/share count;
- tests de splits, negativos, ventanas y outliers.

Gate: golden dataset revisado manualmente y formula publicada en `/metodologia`.

## Fase 4 - Valuacion no financiera V1

Entregables:

- normalizador de statements;
- selector madura/high-growth inicial;
- FCFF multi-etapa, bear/base/bull y sensitivity;
- assumptions workbench, locks y audit trace;
- Damodaran market assumptions snapshots;
- comparison fixtures contra spreadsheets.

Gate: invariantes financieras y tolerancias golden pasan; ninguna llamada LLM es necesaria para recalcular.

## Fase 5 - Valuacion por arquetipo e IA

Entregables incrementales:

1. bancos/aseguradoras: excess return/DDM;
2. ciclicas/commodities: normalizacion de ciclo;
3. perdidas/high growth: revenue-to-margin;
4. REIT: AFFO/NAV;
5. holding/SOTP y distress solo si hay demanda.

Agregar investigacion Tavily, OpenRouter structured assumptions, citations, policy engine y evals.

Gate: set de casos por arquetipo, tasa de rechazo/correccion medida y costo por corrida visible.

## Fase 6 - Argentina y soja

Entregables:

- BCRA v4 y catalogo dinamico;
- datos.gob.ar/INDEC/BNA con snapshots;
- bloques del dashboard y transformaciones reales;
- Rosario/Chicago con contrato, conversion y roll policy;
- alertas de freshness/schema y metodologia.

Gate: series comparadas con ultima publicacion oficial; cambio de schema falla de forma segura.

## Fase 7 - Asistente de preguntas

Entregables:

- acciones acotadas: explicar grafico, comparar empresas, explicar valuacion;
- tool calls solo a funciones internas autorizadas;
- citations y refusal si falta evidencia;
- budgets, tracing y prompt-injection tests.

Gate: eval suite factual/financiera, sin acceso arbitrario a URLs o DB.

## Fase 8 - Cuentas y BYOK

Entregables:

- autenticacion y autorizacion;
- saved views/watchlists/valuations;
- API keys de usuario cifradas con envelope encryption o integracion de vault;
- test/revoke/rotate y redaccion de logs;
- cuotas por usuario/tenant.

Gate: threat model actualizado, aislamiento multi-tenant y prueba de que una key nunca llega al browser tras guardarse.

## Fase 9 - Hardening y lanzamiento

- observabilidad, SLOs, backups y restore drill;
- load/cost tests y rate limiting;
- accessibility/performance audit;
- politica de privacidad, terminos, data licenses y disclaimer;
- runbooks de proveedor caido, parser roto, key comprometida y rollback;
- beta cerrada, feedback y prioridades basadas en uso.

Gate: checklist de produccion firmado y presupuesto/alertas activos.

## Orden del MVP

Un MVP util llega hasta Fase 4. Fases 5 y 6 pueden alternarse segun si el valor principal es valuacion profunda o Argentina. No empezar el chat antes de que datos, calculos y evidencia sean confiables.

## Definicion global de terminado

Una feature esta terminada cuando:

- pregunta y criterio de aceptacion estan claros;
- schema, formula y fuente estan documentados;
- happy path, bordes y degradacion tienen tests;
- datos muestran as-of/source/quality;
- seguridad/licencia fueron revisadas;
- UI es usable por teclado y en movil;
- observabilidad y costo estan considerados;
- preview fue verificado y el diff revisado.

