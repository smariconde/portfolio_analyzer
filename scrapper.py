# SOJA CHICAGO vs ROSARIO - FUENTES OFICIALES - ACTUALIZA AUTOMÁTICO
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
from bcra import get_minorista_exchange_rate # Importar la función para obtener el tipo de cambio

# --- Caching para yfinance ---
CHICAGO_CACHE_FILE = 'chicago_data.csv'
CACHE_DURATION_DAYS = 1 # Cache data for 1 day

def get_chicago_data(cache_file=CHICAGO_CACHE_FILE, cache_days=CACHE_DURATION_DAYS):
    """
    Fetches Chicago soybean data, using a local cache to avoid repeated downloads.
    """
    if os.path.exists(cache_file):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - file_mod_time) < timedelta(days=cache_days):
            print("Cargando datos de Chicago desde el caché...")
            try:
                chicago_series = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze("columns")
                return chicago_series
            except Exception as e:
                print(f"Advertencia: No se pudo leer el archivo de caché '{cache_file}'. Se intentará descargar de nuevo. Error: {e}")

    print("Descargando nuevos datos de Chicago desde yfinance...")
    try:
        chicago = yf.download("ZS=F", period="5y", progress=False)["Close"]
        if chicago.empty:
            raise ValueError("El download de yfinance retornó un DataFrame vacío.")
        chicago.to_csv(cache_file)
        print(f"Datos de Chicago guardados en caché en '{cache_file}'.")
        return chicago
    except Exception as e:
        print(f"Error al descargar datos de Chicago desde yfinance: {e}")
        return pd.Series(dtype=float)

# 1) CHICAGO: yfinance (de Yahoo/CME oficial, USD/ton mensual)
chicago_data = get_chicago_data()
if not chicago_data.empty:
    chicago_usd_ton = chicago_data * 0.3674  # cents/bushel → USD/ton
    chicago_monthly = chicago_usd_ton.resample("MS").mean()  # Mensual
else:
    chicago_monthly = pd.Series(dtype=float) # Inicializar como Serie vacía en caso de error

# 2) ROSARIO: Scrape real de Consiagro (BCR pizarra histórica, ARS/ton)
# Nota: Si falla (anti-scrape), manual de https://www.consiagro.com.ar/files/bd_pizarras_historico.php
url_rosario = 'https://www.consiagro.com.ar/files/bd_pizarra_ros_historico.php' # Corrected URL
tables = pd.read_html(url_rosario)
df_rosario_raw = pd.DataFrame() # Initialize as empty DataFrame
rosario_ars = pd.Series(dtype=float)

if tables:
    df_rosario_raw = tables[0]  # Tabla principal
    if not df_rosario_raw.empty:
        # Ensure there are enough columns before setting names and slicing
        if df_rosario_raw.shape[1] >= 6:
            df_rosario_raw.columns = ['Fecha', 'Trigo', 'Maiz', 'Sorgo', 'Soja_ARS', 'Girasol']
            df_rosario_raw = df_rosario_raw.iloc[1:].copy()
            df_rosario_raw['Fecha'] = pd.to_datetime(df_rosario_raw['Fecha'], format='%Y-%m-%d')
            df_rosario_raw.set_index('Fecha', inplace=True)
            # Limpiar la columna de Soja: remover puntos de miles y reemplazar coma decimal por punto
            soja_ars_cleaned = df_rosario_raw['Soja_ARS'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            rosario_ars_numeric = pd.to_numeric(soja_ars_cleaned, errors='coerce').dropna()
            # Dividir por 100, asumiendo que el precio está en ARS por 100 toneladas
            rosario_ars = rosario_ars_numeric / 100
            rosario_ars = rosario_ars.sort_index() # Ensure index is monotonic
        else:
            print(f"Warning: Rosario data table found but has unexpected number of columns ({df_rosario_raw.shape[1]}). Expected 6. Rosario data will be empty.")
    else:
        print("Warning: Rosario data table found but is empty after initial processing.")
else:
    print("Warning: No tables found at the Rosario data URL. Rosario data will be empty.")

# Convertir ARS a USD usando datos oficiales del BCRA
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
        
        # Alinear el tipo de cambio con las fechas de la soja y rellenar faltantes (fines de semana)
        dolar_aligned = dolar_series.reindex(rosario_ars.index, method='ffill')
        
        # Convertir a USD y eliminar filas sin tipo de cambio
        rosario_usd_daily = (rosario_ars / dolar_aligned).dropna()
        
        # Remuestrear a mensual
        rosario_usd = rosario_usd_daily.resample("MS").mean()
        print("Conversión a USD completada exitosamente usando datos del BCRA.")
    else:
        print("Advertencia: No se pudieron obtener los datos del tipo de cambio del BCRA. La conversión a USD fallará.")
else:
    print("Advertencia: No hay datos de Rosario para convertir a USD.")

# 3) Alinear (a partir de 2024)
start_date = datetime(2024, 1, 1)

df_chicago = chicago_monthly[chicago_monthly.index >= start_date]
# Asegurarse de que df_chicago sea una Serie
if isinstance(df_chicago, pd.DataFrame):
    df_chicago = df_chicago.squeeze()

# Handle case where rosario_usd might be empty
if rosario_usd.empty:
    df = pd.DataFrame({
        'Chicago_USD_ton': df_chicago,
        'Rosario_USD_ton': pd.Series(dtype=float, index=df_chicago.index) # Create an empty Series with matching index
    })
    print("Warning: Rosario data is empty. Plot will only show Chicago data or be incomplete.")
else:
    df_rosario = rosario_usd.loc[rosario_usd.index >= start_date]
    # Asegurarse de que df_rosario sea una Serie
    if isinstance(df_rosario, pd.DataFrame):
        df_rosario = df_rosario.squeeze()
        
    df = pd.DataFrame({
        'Chicago_USD_ton': df_chicago,
        'Rosario_USD_ton': df_rosario
    })
    
    # Solo dropear NaNs si ambas columnas tienen datos, para no eliminar todo si una fuente falla
    if not df['Chicago_USD_ton'].empty and not df['Rosario_USD_ton'].empty:
        df.dropna(inplace=True)


# Check if df is empty after alignment and dropna
if df.empty:
    print("Error: No data available for plotting after alignment. Please check data sources.")
else:
    # 4) Correlación
    corr = df['Chicago_USD_ton'].corr(df['Rosario_USD_ton'])

    # 5) Gráfico PRO
    plt.figure(figsize=(12, 6), dpi=150)
    plt.plot(df.index, df['Chicago_USD_ton'], label='Chicago CBOT (USD/ton)', color='blue', linewidth=2.5)
    plt.plot(df.index, df['Rosario_USD_ton'], label='Rosario Pizarra (USD/ton)', color='red', linewidth=2.5)
    plt.fill_between(df.index, df['Chicago_USD_ton'], df['Rosario_USD_ton'], alpha=0.2, color='gray', label='Diferencia')

    # --- Anotaciones de Eventos Clave ---
    events = [
        ('2025-04-11', 'Fin Cepo Cambiario', 'Anuncio fin del cepo cambiario BCN\nlevanta restricciones al dólar.'),
        ('2025-01-01', 'Entrada EUDR (UE)', 'Regulación UE contra deforestación\nobliga trazabilidad soja.'),
        ('2025-09-22', 'Quita Temporal Retenciones', 'Decreto 682/2025: Retenciones 0%\npara soja.'),
        ('2025-09-25', 'Fin Quita Retenciones', 'Cupo US$7.000M alcanzado\nen 3 días.'),
        ('2025-08-15', 'Pico Exportaciones Soja', 'Agosto 2025: 2do mejor mes exportador\n(demanda China).')
    ]

    # Filtrar eventos que están dentro del rango del gráfico
    min_date, max_date = df.index.min(), df.index.max()
    visible_events = [e for e in events if min_date <= pd.to_datetime(e[0]) <= max_date]
    
    # Lógica para alternar la posición vertical de las anotaciones y evitar solapamiento
    if visible_events:
        y_positions = np.linspace(df['Rosario_USD_ton'].min(), df['Chicago_USD_ton'].max(), len(visible_events) + 2)[1:-1]
        y_pos_iter = iter(sorted(y_positions, reverse=True))

        for date_str, label, full_desc in sorted(visible_events, key=lambda x: pd.to_datetime(x[0])):
            event_date = pd.to_datetime(date_str)
            plt.axvline(x=event_date, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
            
            # Interpolar el valor del precio de Rosario para la fecha del evento
            y_value_rosario = np.interp(event_date.toordinal(), [d.toordinal() for d in df.index], df['Rosario_USD_ton'])
            
            try:
                y_text = next(y_pos_iter)
            except StopIteration:
                y_pos_iter = iter(sorted(y_positions, reverse=True))
                y_text = next(y_pos_iter)

            # Usar el label corto para el título y la descripción completa en el texto
            annotation_text = f"{label}\n({full_desc})"
            
            plt.annotate(annotation_text,
                         xy=(event_date, y_value_rosario),
                         xytext=(event_date + pd.Timedelta(days=5), y_text), # Desplazar texto ligeramente a la derecha
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4, alpha=0.6),
                         fontsize=8, # Reducir tamaño de fuente
                         ha='left', # Alinear a la izquierda
                         va='center',
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=0.5, alpha=0.8))

    # Construir el título dinámicamente para evitar errores si faltan datos
    title = 'Correlación Soja: Chicago vs. Rosario'
    stats = []
    if not df.empty:
        stats.append(f'Pearson: {corr:.3f}')
        if 'Chicago_USD_ton' in df.columns and not df['Chicago_USD_ton'].empty:
            stats.append(f'Actual Chi ${df["Chicago_USD_ton"].iloc[-1]:.0f}')
        if 'Rosario_USD_ton' in df.columns and not df['Rosario_USD_ton'].empty:
            stats.append(f'Actual Ros ${df["Rosario_USD_ton"].iloc[-1]:.0f}')
    if stats:
        title += '\n' + ' | '.join(stats)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('USD por Tonelada')
    plt.xlabel('Fecha')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('soja_correlacion_oficial.png', bbox_inches='tight', dpi=300)
    plt.show()

    print(f'Correlación: {corr:.3f}')
    print('Datos alineados:', len(df))
    print(df.tail())  # Últimos valores
