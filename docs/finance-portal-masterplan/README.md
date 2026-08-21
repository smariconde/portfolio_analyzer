# Masterplan portable: portal financiero inteligente

Este directorio es un kit de arranque para crear, en un repositorio limpio, un portal financiero en Next.js que se despliegue directamente en Vercel. No es una especificacion para copiar la app Streamlit actual: conserva sus mejores ideas y las redisena como un producto web modular, trazable y escalable.

## Resultado buscado

El producto debe responder preguntas concretas:

- Que empresas cumplen determinados ratios y como cambiaron en 2 o 5 anos.
- Cuales de esas empresas tienen CEDEAR activo y cual es su ratio de conversion.
- En que empresas el EPS crecio mas rapido que la capitalizacion de mercado.
- Cuanto vale una empresa bajo un metodo adecuado a su tipo y ciclo de vida.
- Que esta mostrando el regimen macro argentino y que datos sostienen esa lectura.
- Que supuestos, fuentes, fechas y riesgos explican cada resultado.

La IA propone, clasifica, investiga y explica. El codigo deterministico descarga, normaliza, calcula, valida y guarda. Ningun numero de valuacion debe depender de que un LLM haga aritmetica libre.

## Como usar este kit

1. Copiar este directorio a `docs/masterplan/` del repositorio nuevo.
2. Abrir una sesion nueva del agente en la raiz del repo.
3. Pegar el contenido de `00_MASTER_PROMPT.md`.
4. Dejar que el agente complete primero Fase 0 y Fase 1. No pedir todas las fases en un unico cambio.
5. Al final de cada fase, exigir demo, tests, auditoria de datos y un registro de decisiones.

## Mapa de documentos

- `00_MASTER_PROMPT.md`: prompt inicial autocontenido para la sesion especial.
- `01_PRODUCT_AND_SCOPE.md`: producto, usuarios, preguntas, limites y rutas.
- `02_ARCHITECTURE.md`: arquitectura, modulos, contratos y persistencia.
- `03_DATA_AND_PROVENANCE.md`: fuentes, adaptadores, CEDEAR, Argentina y trazabilidad.
- `04_VALUATION_SYSTEM.md`: motor Damodaran, seleccion de metodo, IA y controles.
- `05_UX_UI.md`: experiencia, paginas, graficos y accesibilidad.
- `06_PHASED_ROADMAP.md`: fases, entregables, gates y definicion de terminado.
- `07_AGENT_AND_SKILLS.md`: AGENTS.md, skills externas y skills propias.
- `08_QUALITY_SECURITY_OPERATIONS.md`: pruebas, seguridad, costos y operacion.
- `09_ENVIRONMENT_AND_DEPLOY.md`: variables, setup y despliegue Vercel.
- `10_DECISIONS_AND_SOURCES.md`: decisiones ya tomadas y bibliografia primaria.

## Principios no negociables

1. Fuente y fecha visibles para todo dato importante.
2. Unidades, moneda, periodo fiscal y politica de ajustes forman parte del dato.
3. Formulas puras, versionadas y cubiertas por tests.
4. Los supuestos de IA usan salida estructurada, limites economicos y evidencia.
5. Fallas de un proveedor degradan una seccion; no rompen todo el portal.
6. Secretos solo en servidor, nunca con prefijo `NEXT_PUBLIC_`.
7. El MVP es un monolito modular. Se extrae un servicio solo cuando una metrica real lo justifica.
8. No hay recomendaciones personalizadas ni ejecucion de operaciones.
9. La licencia de redistribucion de cada fuente se valida antes de publicar sus datos.
10. Cada fase termina con una version desplegable.

