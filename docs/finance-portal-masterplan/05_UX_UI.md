# UX/UI y visualizacion

## Direccion

Interfaz financiera sobria, clara y explicable. Evitar tanto el aspecto de planilla cruda como una landing llena de gradientes. La jerarquia es: pregunta, respuesta, evidencia, detalle.

## Sistema visual

- shadcn/ui como base editable, Tailwind y tokens CSS semanticos.
- Tema claro y oscuro desde el inicio, con contraste WCAG AA.
- Tipografia tabular para numeros y una sans legible para texto.
- Verde/rojo nunca son el unico canal: usar icono, signo y etiqueta.
- Formatos por locale `es-AR`, con moneda/unidad siempre visibles.
- Skeletons que conservan layout, estados vacios accionables y errores por modulo.

## Patron de pagina

1. Encabezado con pregunta y contexto.
2. Controles primarios en una sola fila/hoja movil.
3. Respuesta resumida en pocas metric cards.
4. Visual principal.
5. Tabla o drivers.
6. Fuentes, freshness, metodologia y advertencias desplegables.

## Home

- Command/search global por ticker, empresa o CEDEAR.
- Cuatro accesos formulados como preguntas, no como nombres tecnicos.
- Estado de fuentes: actualizado, degradado o requiere configuracion.
- Ultimos analisis solo cuando haya persistencia real.

## Screener

- TanStack Table con virtualizacion solo cuando el volumen la justifique.
- Column picker, orden multiple, filtros tipados y presets.
- Chip `CEDEAR` con tooltip de simbolo BYMA y ratio vigente.
- Columnas con valor actual, 2Y, 5Y y delta; el usuario elige la vista, no se muestran todas juntas en movil.
- Exportar CSV con definiciones, fecha y fuente, no solo numeros.
- Click en fila abre drawer de comparacion rapida; pagina completa para profundidad.

## Fundamental gap

Scatter inicial con Recharts/shadcn:

- X = market cap CAGR; Y = diluted EPS CAGR.
- Linea diagonal `y=x`; tooltip explica que puntos sobre la linea tienen gap positivo.
- Color = sector; borde o icono = CEDEAR; area = market cap actual con escala acotada.
- Toggle 2Y/5Y y filtro sector/CEDEAR/market cap/data quality.
- Lasso no es necesario en MVP; click fija tooltip y abre detalle.
- Tabla sincronizada, ordenada por `fundamental_gap_pp`.
- Panel separado para `loss_to_profit`, `profit_to_loss` y `negative_both`.
- Outliers recortados solo visualmente se etiquetan en el borde; tooltip conserva raw.

No usar ejes log para tasas negativas. Si Recharts pierde rendimiento con el universo real, registrar metrica y migrar ese grafico aislado a ECharts/Canvas mediante dynamic import.

## Valuacion

Wizard sin bloquear a usuarios expertos:

- Paso 1: identidad/fecha/moneda y health de datos.
- Paso 2: arquetipo y metodo sugerido con alternativa.
- Paso 3: assumptions workbench agrupado, con valor sugerido, rango, fuente y lock.
- Paso 4: resultados.

Resultado:

- valor intrinseco/rango, precio y margen de seguridad;
- waterfall de EV a equity;
- barras bear/base/bull;
- sensitivity heatmap WACC/g;
- tabla de flujos por ano;
- driver tornado mas adelante;
- diagnosticos priorizados y source drawer.

La narrativa IA aparece despues de los numeros y cita evidence cards. Un usuario puede recalcular sin volver a llamar a IA cuando cambia supuestos.

## Argentina

- Tabs o secciones por regimen: nominal, monetario, cambiario, actividad, fiscal/externo, agro.
- Cada tarjeta muestra ultimo dato, variacion relevante, fecha y estado.
- Graficos permiten nominal/real, nivel/variacion y ventana temporal cuando la transformacion sea valida.
- `Por que importa` usa texto corto y estable; `Lectura actual` puede ser generada a partir de datos.
- Fechas heterogeneas se muestran por serie; no poner una fecha global falsa.

## Mobile y accesibilidad

- Navegacion inferior o drawer en pantallas pequenas; tablas pasan a cards/columnas prioritarias.
- Todos los graficos tienen titulo, descripcion y tabla accesible equivalente.
- Navegacion completa por teclado, focus visible y targets tactiles suficientes.
- Respetar `prefers-reduced-motion`.
- Tooltips tambien accesibles por focus/click, no solo hover.
- Tests axe y Playwright en flujos criticos.

## Performance budgets

- Server Components por defecto y paquetes de charts cargados solo en paginas que los usan.
- No hidratar tablas o dashboards completos si una isla interactiva alcanza.
- Presupuesto inicial: JS first-load por ruta medido y registrado; ninguna regresion grande sin explicacion.
- Paginacion/consulta server-side para screeners amplios.
- Imagenes y logos con fallback; nunca bloquear contenido financiero por un logo.

