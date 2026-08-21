# Entorno y despliegue Vercel

## Filosofia de configuracion

La app debe arrancar con modulos opcionales deshabilitados y mostrar un setup health claro. Una key faltante no debe causar build failure salvo que sea esencial para el vertical slice elegido.

## Variables

```dotenv
# Publicas, nunca secretos
NEXT_PUBLIC_APP_URL=http://localhost:3000
NEXT_PUBLIC_APP_NAME=Portal Financiero

# Core server-only
DATABASE_URL=
CRON_SECRET=
APP_ENV=development
LOG_LEVEL=info

# Mercado: elegir un provider
MARKET_DATA_PROVIDER=fmp
FMP_API_KEY=
TWELVE_DATA_API_KEY=
FINANCIAL_DATASETS_API_KEY=

# SEC exige identificacion responsable
SEC_USER_AGENT=PortalFinanciero/0.1 contact@example.com

# IA e investigacion
OPENROUTER_API_KEY=
OPENROUTER_MODEL_FAST=
OPENROUTER_MODEL_REASONING=
OPENROUTER_ZDR=true
TAVILY_API_KEY=

# Cache/rate limit opcional hasta hacerlo publico
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=

# Fases posteriores
AUTH_SECRET=
CREDENTIAL_ENCRYPTION_KEY=
SENTRY_DSN=
```

No fijar model IDs en este documento: cambian. La Fase 0 consulta la lista actual, elige modelos compatibles con structured outputs, registra costo/capacidades en ADR y valida las variables.

## Esquema de validacion

Crear grupos:

- `core`: base y cron;
- `marketData`: al menos la key del provider seleccionado;
- `ai`: OpenRouter;
- `research`: Tavily;
- `rateLimit`, `auth`, `observability`: opcionales por fase.

`getModuleHealth()` retorna `ready | degraded | disabled`, missing vars y mensaje seguro. Nunca incluye valores.

## Setup local

```bash
pnpm install
cp .env.example .env.local
pnpm db:migrate
pnpm db:seed:demo
pnpm dev
```

Comandos objetivo:

```bash
pnpm lint
pnpm typecheck
pnpm test
pnpm test:integration
pnpm test:e2e
pnpm build
```

## Vercel

1. Importar repo en Vercel.
2. Provisionar Postgres (Neon/Supabase/otro) desde Marketplace en region compatible.
3. Agregar variables por Development/Preview/Production; marcar keys como sensibles.
4. Ejecutar migraciones mediante job controlado, no implicitamente desde cada Function.
5. Configurar `vercel.json` cron solo para production y proteger con `CRON_SECRET`.
6. Configurar `maxDuration` solo en rutas que lo necesiten; no subirlo globalmente como parche.
7. Desplegar preview, correr E2E smoke y promover.

Ejemplo conceptual:

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "crons": [
    { "path": "/api/cron/daily-market-data", "schedule": "15 3 * * *" },
    { "path": "/api/cron/daily-argentina", "schedule": "45 11 * * *" }
  ]
}
```

Los cron usan UTC. En Hobby solo pueden ejecutarse diariamente y dentro de la hora, por lo que el MVP no depende de precision intrahoraria.

## Check de deployment

- env health sin exponer secretos;
- migracion aplicada y rollback conocido;
- cron auth verificada;
- region DB/Functions documentada;
- provider live smoke de bajo costo;
- source registry/licencia vigente;
- no `NEXT_PUBLIC_*KEY*`;
- logs redacted, rate limit y budget activo si la URL es publica;
- disclaimer/metodologia/freshness visibles.

## BYOK posterior

No guardar keys de usuario en localStorage. Flujo:

1. usuario autenticado envia key por TLS a un Server Action/Route Handler;
2. servidor verifica formato y hace test minimo;
3. cifra con data key y master key externa/rotatable;
4. guarda ciphertext, nonce, provider, key version y ultimos 4;
5. descifra solo durante request server-side autorizada;
6. revoke elimina o inutiliza y audita;
7. logs/traces reciben solo credential ID.

