## Portfolio Analyzer

Distintos scripts para crear portfolios dependiendo de la volatilidad y rendimiento pasados.
Se usan ratios como Sharpe y Sortino para encontrar acciones que sean atractivas para invertir a mediano largo plazo.
El análisis se centra en las acciones del SP500 y en los cedears disponibles en Argentina para poder invertir.

La elección entre el **Sharpe ratio**, el **Sortino ratio** o cualquier otro indicador depende del enfoque y objetivos específicos del portfolio que se desea construir, así como de la tolerancia al riesgo y el comportamiento esperado del activo o activos involucrados. Este es un resumen de cada uno:

### 1. **Sharpe Ratio**
El **Sharpe ratio** mide el rendimiento ajustado al riesgo, es decir, cuánto rendimiento adicional obtienes por cada unidad de riesgo asumido. Se calcula como:

\[
\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
\]

Donde:
- \( R_p \) es el rendimiento del portfolio.
- \( R_f \) es la tasa libre de riesgo (como los bonos del gobierno).
- \( \sigma_p \) es la desviación estándar del rendimiento del portfolio (volatilidad).

- > **1.0**: Buen rendimiento ajustado al riesgo. El portafolio está generando un retorno en exceso razonable en comparación con el riesgo.
- > **2.0**: Muy bueno, indica un portafolio con gran rendimiento en relación con el riesgo asumido.
- > **3.0**: Excelente, se considera óptimo y muestra un portafolio muy eficiente.
- < **1.0**: No es ideal, significa que el portafolio no compensa adecuadamente el riesgo asumido.

**Ventajas**:
- Es una medida de riesgo total, es decir, considera tanto los rendimientos positivos como negativos.
- Ideal para comparar portfolios con el mismo horizonte temporal.

**Desventajas**:
- Penaliza igual tanto la volatilidad positiva (beneficios) como la negativa (pérdidas). Esto puede no ser ideal si estás dispuesto a aceptar más volatilidad a cambio de un mayor rendimiento.

**Cuándo usarlo**: 
- Si deseas comparar portfolios en términos de su rendimiento ajustado al riesgo, y no te importa si el riesgo es negativo o positivo (es decir, no te importa si la volatilidad proviene de grandes ganancias o grandes pérdidas).

---

### 2. **Sortino Ratio**
El **Sortino ratio** es similar al Sharpe ratio, pero tiene una diferencia importante: solo penaliza la volatilidad negativa (pérdidas). Se calcula como:

\[
\text{Sortino Ratio} = \frac{R_p - R_f}{\sigma_{\text{down}}}
\]

Donde \( \sigma_{\text{down}} \) es la desviación estándar de los rendimientos negativos (es decir, la volatilidad en el lado de las pérdidas).

**Ventajas**:
- Penaliza solo la volatilidad negativa, lo que lo hace más adecuado para los inversores que se enfocan en minimizar las pérdidas y no se preocupan por la volatilidad positiva (cuando el valor del portfolio sube).
- Es más relevante para inversores que buscan estabilidad y protección contra caídas.

**Desventajas**:
- Puede ser menos útil en contextos donde la volatilidad positiva también es un riesgo importante (como en mercados con fuertes fluctuaciones de precios).
  
**Cuándo usarlo**: 
- Si estás más preocupado por las pérdidas en lugar de las fluctuaciones generales del portfolio y deseas medir solo el rendimiento ajustado al riesgo en los períodos de caídas.

---

### 3. **Omega Ratio**
El **Omega ratio** mide la relación entre los rendimientos positivos y negativos de un portfolio, proporcionando una visión más completa del rendimiento ajustado al riesgo. Se calcula como:

\[
\text{Omega Ratio} = \frac{\int_{0}^{\infty} (1 - F(x))dx}{\int_{-\infty}^{0} F(x)dx}
\]

Donde \( F(x) \) es la función de distribución acumulada de los rendimientos del portfolio.

**Ventajas**:
- Toma en cuenta toda la distribución de los rendimientos, no solo los rendimientos promedio o la volatilidad.
- No se limita a medidas estadísticas como la desviación estándar.

**Desventajas**:
- Puede ser más complejo de interpretar y calcular que el Sharpe o el Sortino.
  
**Cuándo usarlo**: 
- Si te interesa una medida más detallada de la relación entre los rendimientos positivos y negativos, especialmente si hay rendimientos extremos o eventos no lineales en la distribución.

---

### 4. **Calmar Ratio**
El **Calmar ratio** compara el rendimiento de un activo con el riesgo máximo de caída (drawdown). Se calcula como:

\[
\text{Calmar Ratio} = \frac{R_p}{\text{Máximo Drawdown}}
\]

**Ventajas**:
- Muy útil para evaluar portfolios con una alta exposición al riesgo de pérdidas extremas.
- Se enfoca en el riesgo de colapsos, lo que es muy relevante para inversores que desean minimizar los drawdowns.

**Desventajas**:
- No considera el riesgo total, solo el drawdown.

**Cuándo usarlo**: 
- Si estás buscando minimizar las pérdidas extremas (por ejemplo, durante crisis financieras) y te interesa más la "recuperación" después de un gran desplome que la volatilidad diaria.

---

### 5. **Treynor Ratio**
El **Treynor ratio** mide el rendimiento ajustado al riesgo sistemático (el riesgo no diversificable), considerando solo la beta del portfolio. Se calcula como:

\[
\text{Treynor Ratio} = \frac{R_p - R_f}{\beta_p}
\]

**Ventajas**:
- Ideal para portfolios que están bien diversificados, ya que considera el riesgo sistemático.
- Se centra en el rendimiento extra obtenido por cada unidad de riesgo sistemático.

**Desventajas**:
- Solo es relevante para portfolios bien diversificados. Si el portfolio no es muy diversificado, su beta podría no ser representativa.

**Cuándo usarlo**: 
- Si tu portfolio está bien diversificado y deseas evaluar el rendimiento en función del riesgo sistemático, no de la volatilidad total.

---

### ¿Qué Ratio Deberías Usar?
- **Si te interesa un rendimiento ajustado al riesgo total** y quieres un análisis simple y efectivo, **el Sharpe ratio** es adecuado.
- **Si te preocupa más el riesgo a la baja**, o las pérdidas extremas, **el Sortino ratio** es una mejor opción.
- **Si estás enfocado en la minimización de drawdowns**, entonces el **Calmar ratio** podría ser útil.
- Si tienes un portfolio muy diversificado y te interesa cómo se comporta respecto al mercado en general, el **Treynor ratio** es la opción indicada.

### Resumen:
- **Sharpe Ratio**: Para riesgo total, ideal si te interesa una medida general del rendimiento ajustado al riesgo.
- **Sortino Ratio**: Para minimizar la volatilidad negativa, adecuado si te preocupan más las pérdidas que la volatilidad general.
- **Omega Ratio**: Para una evaluación más completa de la distribución de los rendimientos.
- **Calmar Ratio**: Para evaluar la protección contra pérdidas extremas (drawdowns).
- **Treynor Ratio**: Para portfolios bien diversificados, enfocado en el riesgo sistemático.
