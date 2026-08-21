# Producto y alcance

## Propuesta

Un portal de decision financiera que combina datos reproducibles, herramientas cuantitativas y explicaciones asistidas por IA. No intenta ser una terminal de trading ni un agregador de noticias. Cada modulo debe responder una pregunta y mostrar como llego a la respuesta.

## Usuarios iniciales

- Inversor argentino que analiza acciones globales accesibles mediante CEDEAR.
- Analista fundamental que quiere filtrar, comparar y valorar empresas.
- Usuario que sigue el regimen macro argentino y su impacto en activos.
- Desarrollador/owner que opera una instancia propia con sus API keys en Vercel.

## Jobs to be done

1. Encontrar: "mostrame CEDEARs de tecnologia con ROIC alto, deuda moderada y mejora de margen en 5 anos".
2. Comparar: "como evolucionaron EPS, ventas, margen, capitalizacion y acciones diluidas en 2/5 anos".
3. Detectar divergencias: "donde crecieron mas las ganancias que el valor de mercado".
4. Valorar: "que metodo corresponde, cuales son los supuestos y que rango de valor resulta".
5. Entender Argentina: "que dicen reservas, liquidez, inflacion, actividad, sector externo y soja".
6. Auditar: "de donde salio cada dato, de que fecha es y que transformacion recibio".

## Navegacion objetivo

- `/`: preguntas frecuentes, buscador global, estado de datos y accesos rapidos.
- `/empresas`: screener por sector, CEDEAR, ratio, periodo y metrica.
- `/empresas/[symbol]`: ficha, series fundamentales, filings y trazabilidad.
- `/oportunidades/fundamental-gap`: market cap vs EPS para 2 y 5 anos.
- `/valuacion`: selector/buscador y valuaciones recientes.
- `/valuacion/[symbol]`: wizard automatico, supuestos, escenarios y resultados.
- `/argentina`: tablero por bloques, no una pared de graficos.
- `/metodologia`: formulas, fuentes, cobertura, limitaciones y changelog.
- `/configuracion`: health de proveedores en MVP; cuentas/BYOK mas adelante.

## Modulos

### 1. Empresas y ratios

- Universo inicial: subyacentes de CEDEAR de acciones, mas un universo configurable de empresas de EE.UU.
- Seleccion de 2/5 anos y TTM/annual cuando corresponda.
- Constructor de filtros con AND/OR limitado, presets guardables mas adelante.
- Ratios iniciales: P/E, EV/EBITDA, EV/Sales, P/B, FCF yield, ROIC, ROE, gross/operating margin, net debt/EBITDA, revenue/EPS/FCF CAGR y share-count CAGR.
- La definicion de cada ratio es unica y versionada; no mezclar formulas de proveedores sin normalizacion.

### 2. Fundamental gap

El scatter compara crecimiento anualizado de market cap y EPS. Es una herramienta de deteccion, no una senal de compra. El detalle debe exponer recompras/dilucion, punto de partida ciclico, extraordinarios y EPS no comparable.

### 3. Valuacion

Flujo progressive disclosure:

1. Elegir empresa y fecha.
2. Mostrar datos normalizados y faltantes.
3. Clasificar arquetipo/metodo con explicacion.
4. Proponer supuestos con evidencia.
5. Permitir editar y bloquear supuestos.
6. Calcular escenarios/sensibilidad.
7. Explicar drivers, riesgos y calidad.
8. Guardar un snapshot reproducible.

### 4. Argentina

Bloques por pregunta:

- Nominalidad: inflacion, REM, tasas, CER/UVA.
- Moneda y liquidez: base, M2, depositos, credito y tasas reales.
- Dolares y competitividad: oficial, brechas con fuentes licenciadas, ITCRM y reservas.
- Actividad e ingresos: EMAE, industria/construccion cuando haya fuente estable, salarios reales.
- Fiscal y externo: resultado primario/financiero, comercio exterior y terminos de intercambio.
- Agro: soja Rosario, referencia Chicago, basis/spread, percentil historico y alertas de calidad.

Cada bloque termina con "por que importa", "que cambio" y "que podria invalidar esta lectura". La IA resume datos ya calculados; no crea series.

## Fuera de alcance inicial

- Ejecucion de ordenes, conexion a broker o custodia.
- Recomendaciones personales basadas en patrimonio o tolerancia al riesgo.
- Portfolios sociales, copy trading, pagos y suscripciones.
- Datos tick-by-tick o promesa de tiempo real.
- Chat general que responda sin fuentes.
- RAG/vector database antes de demostrar una necesidad real.
- App movil nativa.

## Metricas de producto

- Porcentaje de resultados con fuente, fecha y unidad completas: objetivo 100%.
- Porcentaje de valuaciones reproducibles con el mismo snapshot/version: 100%.
- Tiempo hasta primer insight util desde el buscador: menos de 60 segundos.
- Tasa de fallas de ingesta y antiguedad por fuente.
- Costo medio de APIs/IA por valuacion y por usuario activo.
- Porcentaje de propuestas de IA corregidas/rechazadas por policy engine.
- Uso de filtros, exportaciones y comparaciones, no solo page views.

## Politica editorial y legal

- Mostrar "informacion educativa, no asesoramiento financiero" sin usarlo como excusa para baja calidad.
- Separar hechos, transformaciones, supuestos y opinion generada.
- No usar "barata", "comprar" o "vender" como verdad; usar escenarios y evidencia.
- Registrar terminos/licencia/redistribucion de cada proveedor antes del lanzamiento publico.

