import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Import matplotlib.dates
import numpy as np
from datetime import datetime, timedelta
import os
# Asegúrate de que 'bcra' esté disponible, si no, necesitarás instalarla o usar una alternativa.
try:
    from bcra import get_minorista_exchange_rate
except ImportError:
    print("Advertencia: No se pudo importar 'bcra.get_minorista_exchange_rate'. Asumiendo que el tipo de cambio se manejará con una fuente alternativa o se llenará con NaNs.")
    # Función dummy en caso de que 'bcra' no esté disponible
    def get_minorista_exchange_rate(fecha_desde, fecha_hasta):
        print(f"Usando función dummy para tipo de cambio. Buscando en BCRA desde {fecha_desde} hasta {fecha_hasta} falló.")
        return None 

# --- Caching para yfinance ---
CHICAGO_CACHE_FILE = 'chicago_data.csv' # Nuevo nombre de caché para datos diarios
CACHE_DURATION_DAYS = 1

def get_chicago_data(cache_file=CHICAGO_CACHE_FILE, cache_days=CACHE_DURATION_DAYS):
    """
    Fetches Chicago soybean data, using a local cache to avoid repeated downloads.
    Fetches data for a longer period to ensure daily data is available from start_date.
    """
    if os.path.exists(cache_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - file_mod_time) < timedelta(days=cache_days):
            print("Cargando datos de Chicago desde el caché...")
            try:
                # Use 'Date' as index name for consistency if the original file had it
                chicago_series = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze("columns")
                return chicago_series
            except Exception as e:
                print(f"Advertencia: No se pudo leer el archivo de caché '{cache_file}'. Se intentará descargar de nuevo. Error: {e}")

    print("Descargando nuevos datos de Chicago desde yfinance (diario, 5 años)...")
    try:
        # Descargamos los datos diarios (sin resamplear a mensual)
        chicago = yf.download("ZS=F", period="5y", progress=False)["Close"]
        if chicago.empty:
            raise ValueError("El download de yfinance retornó un DataFrame vacío.")
        chicago.to_csv(cache_file)
        print(f"Datos de Chicago guardados en caché en '{cache_file}'.")
        return chicago
    except Exception as e:
        print(f"Error al descargar datos de Chicago desde yfinance: {e}")
        return pd.Series(dtype=float)

# 1) CHICAGO: yfinance (de Yahoo/CME oficial, USD/ton diario)
chicago_data = get_chicago_data()
if not chicago_data.empty:
    chicago_usd_ton = chicago_data * 0.3674  # cents/bushel → USD/ton
    # Mantenemos la granularidad diaria
else:
    chicago_usd_ton = pd.Series(dtype=float)

# 2) ROSARIO: Scrape real de Consiagro (BCR pizarra histórica, ARS/ton)
url_rosario = 'https://www.consiagro.com.ar/files/bd_pizarra_ros_historico.php'
tables = pd.read_html(url_rosario)
rosario_ars = pd.Series(dtype=float)

if tables:
    df_rosario_raw = tables[0]
    if not df_rosario_raw.empty and df_rosario_raw.shape[1] >= 6:
        df_rosario_raw.columns = ['Fecha', 'Trigo', 'Maiz', 'Sorgo', 'Soja_ARS', 'Girasol']
        df_rosario_raw['Fecha'] = pd.to_datetime(df_rosario_raw['Fecha'], format='%Y-%m-%d')
        df_rosario_raw.set_index('Fecha', inplace=True)
        soja_ars_cleaned = df_rosario_raw['Soja_ARS'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        rosario_ars_numeric = pd.to_numeric(soja_ars_cleaned, errors='coerce').dropna()
        # Asumo que la corrección del precio fue: ARS/100 ton → ARS/ton
        rosario_ars = rosario_ars_numeric / 100 
        rosario_ars = rosario_ars.sort_index()
    else:
        print("Warning: Rosario data table found but has unexpected columns or is empty.")
else:
    print("Warning: No tables found at the Rosario data URL. Rosario data will be empty.")

# Convertir ARS a USD usando datos oficiales del BCRA (diario)
rosario_usd = pd.Series(dtype=float)
if not rosario_ars.empty:
    start_date_bcra = rosario_ars.index.min().strftime('%Y-%m-%d')
    end_date_bcra = rosario_ars.index.max().strftime('%Y-%m-%d')
    
    print(f"Obteniendo tipo de cambio del BCRA desde {start_date_bcra} hasta {end_date_bcra}...")
    dolar_data = get_minorista_exchange_rate(fecha_desde=start_date_bcra, fecha_hasta=end_date_bcra)
    
    if dolar_data:
        dolar_df = pd.DataFrame(dolar_data)
        dolar_df['fecha'] = pd.to_datetime(dolar_df['fecha'])
        dolar_df.set_index('fecha', inplace=True)
        dolar_series = dolar_df['valor']
        
        # Alinear el tipo de cambio con las fechas de la soja y rellenar faltantes
        dolar_aligned = dolar_series.reindex(rosario_ars.index, method='ffill')
        
        # Convertir a USD y eliminar filas sin tipo de cambio
        rosario_usd = (rosario_ars / dolar_aligned).dropna()
        print("Conversión a USD completada exitosamente usando datos del BCRA.")
    else:
        print("Advertencia: No se pudieron obtener los datos del tipo de cambio del BCRA. La conversión a USD fallará.")

# 3) Alinear y Filtrar (Diario, desde Enero 2024)
start_date = datetime(2024, 1, 1) # <--- CAMBIO CLAVE A ENERO 2024

df_chicago_daily = chicago_usd_ton[chicago_usd_ton.index >= start_date].squeeze()
df_rosario_daily = rosario_usd.loc[rosario_usd.index >= start_date].squeeze()

# Alinear ambas series. Usaremos el índice de Chicago como referencia
# y rellenaremos los datos faltantes de Rosario con el último valor conocido.
df_combined = pd.DataFrame({
    'Chicago_USD_ton': df_chicago_daily,
}).copy()
# Reindexar Rosario al índice de Chicago y rellenar hacia adelante
df_combined['Rosario_USD_ton'] = df_rosario_daily.reindex(df_combined.index, method='ffill')

# Advertir si los datos de Rosario no están actualizados
if not df_rosario_daily.empty and not df_chicago_daily.empty:
    last_chicago_date = df_chicago_daily.index.max()
    last_rosario_date = df_rosario_daily.index.max()
    if last_rosario_date < last_chicago_date:
        print("\n" + "="*50)
        print(f"⚠️ ADVERTENCIA: Datos de Rosario desactualizados.")
        print(f"   Último dato Chicago: {last_chicago_date.strftime('%Y-%m-%d')}")
        print(f"   Último dato Rosario: {last_rosario_date.strftime('%Y-%m-%d')}")
        print("   → Se rellenarán los días faltantes con el último valor conocido de Rosario.")
        print("="*50)

# Eliminar filas iniciales si Rosario aún no tenía datos (antes del ffill)
df_combined.dropna(inplace=True)

df = df_combined.copy() # Usaremos 'df' para el análisis final

# Check if df is empty
if df.empty:
    print("Error: No data available for plotting after alignment and filtering from 2024-01-01. Please check data sources and date range.")
else:
    # 4) Correlación Diaria
    corr = df['Chicago_USD_ton'].corr(df['Rosario_USD_ton'])

    # 5) Análisis de la Brecha (Spread = Chicago - Rosario)
    df['Spread_USD'] = df['Chicago_USD_ton'] - df['Rosario_USD_ton']
    df['Spread_Pct'] = (df['Spread_USD'] / df['Chicago_USD_ton']) * 100

    # Estadísticas Clave
    last_date = df.index[-1].strftime('%Y-%m-%d')
    brecha_usd_actual = df['Spread_USD'].iloc[-1]
    brecha_pct_actual = df['Spread_Pct'].iloc[-1]
    
    brecha_usd_avg = df['Spread_USD'].mean()
    
    # Análisis de la evolución de la brecha (Tendencia, basado en Porcentaje)
    last_30_days_avg_pct = df['Spread_Pct'].tail(30).mean()
    
    # Determinar el primer mes de datos disponibles
    first_month_end = df.index[0] + timedelta(days=30)
    first_month_avg_pct = df['Spread_Pct'][df.index <= first_month_end].mean()
    
    tendencia_str = "N/A" # Valor por defecto
    if not np.isnan(last_30_days_avg_pct) and not np.isnan(first_month_avg_pct) and first_month_avg_pct != 0:
        # Un cambio negativo en el spread % significa que se achicó (lo cual es bueno)
        cambio_brecha_pct_evol = ((last_30_days_avg_pct - first_month_avg_pct) / abs(first_month_avg_pct)) * 100
        evolucion_brecha = "achicamiento" if cambio_brecha_pct_evol < 0 else "ampliación"
        evolucion_str = f"La brecha ha tenido un {evolucion_brecha} del {abs(cambio_brecha_pct_evol):.1f}%"
        tendencia_str = f"{evolucion_brecha} del {abs(cambio_brecha_pct_evol):.1f}%"
        evolucion_msg = f" (Avg % Últimos 30 días vs. Avg % Primer Mes)."
    else:
        evolucion_str = "No se pudo calcular la evolución de la brecha (datos insuficientes)."
        evolucion_msg = ""


    # --- Resultados en Consola ---
    print("\n" + "="*50)
    print("📊 ANÁLISIS DIARIO SOJA (Desde 2024-01-01)")
    print("="*50)
    print(f"✅ Correlación Diaria (Chicago vs. Rosario): {corr:.4f} (Muy Alta)")
    print("-" * 50)
    print(f"🗓️ Fecha del Último Dato: {last_date}")
    print(f"💰 Chicago Actual: USD {df['Chicago_USD_ton'].iloc[-1]:.2f} / ton")
    print(f"🇦🇷 Rosario Actual: USD {df['Rosario_USD_ton'].iloc[-1]:.2f} / ton")
    print("-" * 50)
    print("📉 ANÁLISIS DE LA BRECHA PORCENTUAL (Spread % = (Chicago - Rosario) / Chicago)")
    print(f"⭐ Brecha Porcentual Actual: {brecha_pct_actual:.2f}%")
    print(f"➡️ Brecha Porcentual Promedio (Desde Ene 2024): {df['Spread_Pct'].mean():.2f}%")
    print(f"➡️ Brecha Porcentual Promedio (Últimos 30 días): {last_30_days_avg_pct:.2f}%")
    print(f"📈 Evolución de la Brecha: {evolucion_str}{evolucion_msg}")
    print("="*50 + "\n")


    # 6) Gráfico de Precios (Diario) - Estilo Bloomberg
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150) # Usar fig, ax para un control más fino

    # Líneas de precios
    ax.plot(df.index, df['Chicago_USD_ton'], label='Chicago CBOT (USD/ton)', color='#0077b6', linewidth=1.8) # Azul Bloomberg
    ax.plot(df.index, df['Rosario_USD_ton'], label='Rosario Pizarra (USD/ton)', color='#d62828', linewidth=1.8) # Rojo Bloomberg

    # Relleno entre líneas (Brecha visual)
    ax.fill_between(df.index, df['Chicago_USD_ton'], df['Rosario_USD_ton'], alpha=0.1, color='gray')

    # --- Anotaciones de Eventos Clave ---
    events = [
        ('2024-03-18', 'Cosecha gruesa 2024', 'Primera máquina: San Francisco'),
        ('2025-03-24', 'Cosecha gruesa 2025', 'Post-feriado Semana Santa'),
        ('2025-04-11', 'Fin Cepo Cambiario', 'Anuncio fin del cepo cambiario BCRA'),
        ('2025-09-22', 'Quita Temporal Retenciones', 'Decreto 682/2025: Retenciones 0%'),
        ('2025-08-15', 'Pico Exportaciones Soja', 'Agosto 2025, demanda China')
    ]

    # Filtrar eventos que están dentro del rango del gráfico
    min_date, max_date = df.index.min(), df.index.max()
    visible_events = [e for e in events if min_date <= pd.to_datetime(e[0]) <= max_date]
    
    # Lógica para posicionar anotaciones verticalmente y sin flechas
    if visible_events:
        # Generar posiciones Y distribuidas uniformemente en el rango del gráfico
        y_range = df['Chicago_USD_ton'].max() - df['Rosario_USD_ton'].min()
        y_positions = np.linspace(df['Rosario_USD_ton'].min() + y_range * 0.1, df['Chicago_USD_ton'].max() - y_range * 0.1, len(visible_events) + 2)[1:-1]
        y_pos_iter = iter(sorted(y_positions, reverse=True)) # Iterar de arriba hacia abajo

        for date_str, label, full_desc in sorted(visible_events, key=lambda x: pd.to_datetime(x[0])):
            event_date = pd.to_datetime(date_str)
            
            # Línea vertical para el evento
            ax.axvline(x=event_date, color='gray', linewidth=0.8, alpha=0.6)
            
            # Obtener un valor Y aproximado para la anotación (usando el valor de Rosario como referencia base)
            try:
                y_value_ref = df.loc[event_date, 'Rosario_USD_ton']
            except KeyError:
                # Si la fecha exacta no está, interpolar o usar el punto más cercano
                y_value_ref = np.interp(event_date.toordinal(), [d.toordinal() for d in df.index], df['Rosario_USD_ton'])

            # Obtener posición Y para el texto de la anotación
            try:
                y_text = next(y_pos_iter)
            except StopIteration:
                y_pos_iter = iter(sorted(y_positions, reverse=True)) # Reiniciar iterador si se acaban las posiciones
                y_text = next(y_pos_iter)

            annotation_text = f"{label}\n({full_desc})"
            
            # Anotación sin flecha, solo texto con fondo
            ax.text(event_date - pd.Timedelta(days=18), y_text, annotation_text, 
                    rotation=90,
                    fontsize=5.5, # Tamaño de fuente más pequeño para anotaciones
                    ha='left', # Alineación horizontal a la izquierda
                    va='center', # Alineación vertical al centro
                    bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#cccccc", lw=0.5, alpha=0.9)) # Fondo claro y borde sutil

    # Título y subtítulo estilo Bloomberg
    title_price = f'Precios Diarios de Soja: Chicago vs. Rosario'
    subtitle_price = f'Periodo: Desde {start_date.year} | Correlación Pearson: {corr:.4f} | Última Brecha: ${brecha_usd_actual:.2f} ({brecha_pct_actual:.2f}%)'

    ax.set_title(title_price, fontsize=14, fontweight='bold', loc='left', pad=20) # Título a la izquierda
    ax.text(0.01, 0.95, subtitle_price, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor='#f8f9fa', alpha=0.95)) # Subtítulo como caja de texto

    ax.set_ylabel('USD por Tonelada', fontsize=11)
    ax.set_xlabel('Fecha', fontsize=11)
    
    # Leyenda en la parte inferior izquierda
    ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black', fontsize=9)
    
    # Grid más sutil
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Formato de fechas en el eje X
    # Usar AutoDateLocator y AutoDateFormatter para manejar fechas de manera más robusta
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10) # Use AutoDateLocator
    formatter = mdates.AutoDateFormatter(locator) # Use AutoDateFormatter
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    # Ajustes generales
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # Ajustar para el suptitle y dejar espacio arriba
    plt.subplots_adjust(top=0.88) # Ajustar espacio superior para el subtítulo

    plt.savefig('soja_precios_diario_2024.png', bbox_inches='tight', dpi=300)
    plt.show()

    # 7) Gráfico de la Brecha (Diario, en Porcentaje)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    ax.plot(df.index, df['Spread_Pct'], 
            color='#2c3e50', linewidth=1.2, alpha=0.7, label='Brecha diaria')

    # EMA 20 días (tendencia estructural)
    ema20 = df['Spread_Pct'].ewm(span=20, adjust=False).mean()
    ax.plot(df.index, ema20, color='#e74c3c', linewidth=2.2, 
            label='EMA 20 días')

    # Promedio histórico (2024-2025) — línea punteada
    hist_mean = df['Spread_Pct'].mean()
    ax.axhline(hist_mean, color='black', linestyle='--', linewidth=1.8, 
            label=f'Promedio histórico: {hist_mean:+.2f}%')

    # # Banda ±1σ móvil
    # rolling_mean = df['Spread_Pct'].rolling(30).mean()
    # rolling_std  = df['Spread_Pct'].rolling(30).std()
    # upper = rolling_mean + rolling_std
    # lower = rolling_mean - rolling_std
    # ax.fill_between(df.index, lower, upper, color='#3498db', alpha=0.12)

    # Línea cero
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.8)

    # Último punto destacado
    last_date = df.index[-1]
    last_spread = df['Spread_Pct'].iloc[-1]
    last_ema = ema20.iloc[-1]
    ax.scatter([last_date], [last_spread], color='#e74c3c', s=10, zorder=10)
    ax.annotate(f'{last_spread:+.2f}%',
                xy=(last_date, last_spread),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold', color='#e74c3c',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e74c3c", linewidth=1.5))

    # Título y subtítulo
    ax.set_title('Brecha Soja: Rosario vs. Chicago\n'
                'Diferencia porcentual diaria tomando USD oficial',
                fontsize=15, fontweight='bold', pad=20)
    ax.text(0.01, 0.95,
            f'Última sesión: {last_date.strftime("%d %b %Y")} | '
            f'Brecha: {last_spread:+.2f}% | EMA20: {last_ema:+.2f}% | '
            f'Precios: {rosario_ars[-1]:.0f} Ars/{chicago_usd_ton[-1]:.2f} Usd | Oficial: {dolar_aligned[-1]}',
            transform=ax.transAxes, fontsize=10.5, verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor='#f8f9fa', alpha=0.95))

    # Ejes
    ax.set_ylabel('Brecha (%)', fontsize=12)
    ax.set_xlabel('Fecha', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Leyenda: inferior izquierda
    ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black', fontsize=10)

    # Formato de fechas
    ax.xaxis.set_major_formatter(plt.FixedFormatter(
        df.index.strftime('%b %Y').unique()[::3]))
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    # Márgenes
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)

    # 1. ZONAS DE DECISIÓN (colores institucionales)
    zona_vender   = 20.0   # < 20 % → Rosario caro
    zona_esperar  = 25.0   # 20-25 % → neutral
    zona_guardar  = 28.0   # > 28 % → Rosario barato

    ax.axhspan(0, zona_vender,   color='#27ae60', alpha=0.08, label='Venta agresiva')
    ax.axhspan(zona_vender, zona_esperar, color='#f39c12', alpha=0.08, label='Venta selectiva')
    ax.axhspan(zona_esperar, 40, color='#c0392b', alpha=0.08, label='Acumular')

    # 2. EVENTOS HISTÓRICOS (explican picos/bajadas)
    eventos = {
        "2024-03-18": "Cosecha gruesa 2024",
        '2024-07-11': 'Blend 80/20 - soja 26 % (baja desde 33 %)',
        '2024-11-01': 'Fin Blend',
        "2025-01-28": "DNU 38/2025 oficializa retenciones 26%",
        "2025-03-24": "Cosecha gruesa 2025",
        "2025-06-24": "Pico DJVE récord US$6.500M",
        "2025-07-28": "Milei anuncia retenciones permanentes 26%",
        "2025-09-22": "Retenciones 0% temporal",
    }

    for fecha_str, texto in eventos.items():
        try:
            fecha = pd.to_datetime(fecha_str)
            if fecha in df.index:
                y = df.loc[fecha, 'Spread_Pct']
                ax.axvline(fecha, color='gray', linewidth=0.8, alpha=0.6)
                # Mover el bbox hacia la izquierda de la línea (ajusta shift_days según necesites)
                shift_days = 8
                shifted_x = fecha - pd.Timedelta(days=shift_days)
                ax.text(shifted_x, 0.02, texto,
                        rotation=90, fontsize=5.5, alpha=0.9,
                        ha='right', va='bottom',
                        transform=ax.get_xaxis_transform(),
                        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#cccccc", lw=0.5, alpha=0.9))
        except:
            continue

    # Guardar
    plt.savefig('soja_brecha_final.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Gráfico final generado: soja_brecha_final.png")
    print(f"   → Brecha {last_date.strftime('%d/%m/%Y')}: {last_spread:+.2f}%")
    print(f"   → vs. promedio histórico: {last_spread - hist_mean:+.2f} pp")
