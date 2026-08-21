# Calidad, seguridad y operacion

## Piramide de pruebas

### Dominio unitario

- CAGR, TTM, averages, ratios, FX/unit conversion y fiscal alignment.
- FCFF/DDM/excess return, bridge EV-equity, terminal value y sensitivity.
- Casos: null, cero, negativos, NaN/Infinity, restatement, split y currency mismatch.
- `fast-check` para invariantes financieras y rangos amplios.

### Contract/provider

- Zod valida fixtures reales sanitizadas.
- Parser conserva raw/provenance y rechaza schema inesperado.
- 429/5xx/timeout, paginacion, empty response y partial data.
- Sin red en unit tests; recordings solo si licencia permite guardarlas.

### Integracion

- migraciones contra Postgres real en CI;
- repositorios, ingestion run atomic y dedupe/idempotencia;
- Route Handlers con auth/rate-limit/error envelope.

### E2E

- buscar empresa, filtrar CEDEAR, abrir detail;
- alternar 2Y/5Y en fundamental gap;
- editar supuesto y recalcular sin IA;
- proveedor caido y estado degraded;
- navegacion movil/teclado y axe.

### Evals IA

- schema compliance y evidence IDs validos;
- selector de metodo en un set estratificado;
- no inventar datos/citas;
- prompt injection en filings/web;
- abstencion cuando falta evidencia;
- costo/tokens/latencia y policy correction rate.

No testear prosa exacta. Testear hechos, estructura, citas y limites.

## CI

Orden rapido a lento:

1. format/lint;
2. typecheck;
3. unit/property;
4. build;
5. integration;
6. Playwright smoke;
7. dependency/secrets scan.

Nightly: contract tests live de bajo volumen, data freshness, full E2E y evals IA controladas. Live tests no bloquean un PR por una caida externa sin diagnostico.

## Threat model inicial

- Robo de API keys server-owned o BYOK.
- Abuso de endpoints costosos/IA.
- Prompt injection desde web, filings o nombres de empresas.
- SSRF mediante URLs entregadas a extractores.
- SQL injection y filtros de screener no acotados.
- Cache poisoning o cross-tenant leak.
- Exfiltracion por logs/traces/error messages.
- Supply-chain de npm y skills.
- Datos financieros falsos por parser roto o fuente comprometida.

## Controles

- Secrets privados/sensibles de Vercel, `server-only` y redaccion de logs.
- CSP, headers seguros, validation Zod y DTO minimization.
- Rate limit por IP/usuario/accion; quotas y max cost por request.
- Allowlist de dominios para research financiero; bloquear IPs privadas/redirections sospechosas.
- Herramientas del agente con argumentos tipados y sin SQL/URL arbitrarios.
- OpenRouter model/provider allowlist, budget guardrail, ZDR cuando corresponda.
- `CRON_SECRET` y endpoints idempotentes; lock en DB contra corridas concurrentes.
- Dependabot/Renovate con PR, lockfile, audit y pin de GitHub Actions a SHA.
- Skills externas auditadas y versionadas.
- En BYOK: envelope encryption, key versioning, masked display, revoke/rotate y nunca devolver plaintext.

## Observabilidad

Logs JSON con:

- `request_id`, `user_id_hash`, `module`, `provider`, `operation`;
- status, latency, retries, cache hit, rows, freshness;
- AI model, prompt version, tokens, estimated cost y policy outcome;
- valuation/input/engine version IDs.

Nunca loggear prompts completos por defecto, keys, headers auth o filings privados del usuario. Agregar error tracking y tracing cuando el vertical slice este estable.

## SLOs iniciales

- Paginas cacheadas: 99.5% disponibilidad mensual objetivo.
- Lecturas comunes p95 server < 1.5 s sin contar streaming externo.
- Ingestas oficiales diarias exitosas o alerta antes de 24 h de atraso.
- 100% de valuaciones con version/input/evidence trace.
- Error budget separado por proveedor; degradacion parcial no cuenta como outage total.

## Costo

Medir desde Fase 1:

- invocaciones/duracion/CPU de Vercel;
- storage/egress/queries Postgres;
- requests y creditos por proveedor;
- tokens/costo por modelo y feature;
- cache hit ratio.

Limites: max empresas por screener, max documentos/evidence, max steps de IA, timeouts y circuit breaker. No refrescar universos completos desde una request de usuario.

## Runbooks necesarios

- proveedor 429/caido;
- parser detecta cambio de schema;
- datos stale o discrepancia entre fuentes;
- cron duplicado/fallido;
- migracion fallida/rollback;
- key comprometida/rotacion;
- gasto IA anormal;
- valuacion materialmente incorrecta y retraction/audit.

## Backups y cambios de metodologia

- Backups y restore drill de Postgres antes de launch.
- Valuation runs y source snapshots son append-only salvo datos personales.
- Cambiar formula/metodo incrementa version y changelog; no reescribir historico.
- Correcciones materiales identifican corridas afectadas y permiten recalculo.

