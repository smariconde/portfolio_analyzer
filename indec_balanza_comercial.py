# -*- coding: utf-8 -*-
"""
Script para obtener, procesar y graficar datos de la balanza comercial de Argentina.
Fuente de datos: Archivo Excel publicado por el INDEC.

Versión mejorada: Obtiene la URL de un archivo Excel estable del INDEC
y procesa datos con encabezados multinivel.
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates # Importar para manejo de fechas en el eje X
from io import BytesIO # Para leer el contenido binario del Excel
# No se necesita BeautifulSoup ni re si la URL es estable

def obtener_y_graficar_balanza_comercial_excel():
    """
    Función principal para obtener, procesar y graficar los datos
    de la balanza comercial argentina desde un archivo Excel del INDEC.
    """
    # 1. Definir la URL del archivo Excel del INDEC (URL estable proporcionada por el usuario)
    EXCEL_URL = "https://www.indec.gob.ar/ftp/cuadros/economia/balanmensual.xls"

    print(f"Intentando descargar el archivo Excel desde: {EXCEL_URL}")

    # 2. Descargar el archivo Excel
    try:
        response = requests.get(EXCEL_URL, timeout=30) # Timeout de 30 segundos
        response.raise_for_status() # Lanza una excepción para códigos de error HTTP
        excel_content = BytesIO(response.content)
        print("Archivo Excel descargado exitosamente.")
    except requests.exceptions.HTTPError as http_err:
        print(f"Error HTTP al descargar el archivo Excel: {http_err}")
        print(f"URL intentada: {EXCEL_URL}")
        print("Verifica que la URL sea correcta y accesible, y que el archivo exista.")
        return
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Error de conexión al descargar el archivo Excel: {conn_err}")
        return
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout al descargar el archivo Excel: {timeout_err}")
        return
    except requests.exceptions.RequestException as req_err:
        print(f"Error genérico de requests al descargar el archivo: {req_err}")
        return

    # 3. Leer y procesar el archivo Excel con pandas
    print("Procesando archivo Excel...")
    try:
        # Leer la hoja 'FOB-CIF' con encabezados en las filas 3 y 4 (índices 2 y 3)
        df = pd.read_excel(excel_content, sheet_name='FOB-CIF', header=[2, 3])

        # Limpiar y renombrar columnas
        # Las columnas serán un MultiIndex. Necesitamos seleccionar 'Total mensual'
        # y renombrar 'Período'
        
        # Aplanar el MultiIndex de columnas para facilitar el acceso
        # Se unen los niveles del MultiIndex con un guion bajo, por ejemplo: 'Exportaciones_Total mensual'
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
        
        print("DEBUG: Columnas después de aplanar MultiIndex:")
        print(df.columns.tolist())
        print("DEBUG: df.head() después de aplanar MultiIndex:")
        print(df.head().to_string())

        # Renombrar las columnas clave
        # Based on debug output, the year is in 'Unnamed: 0_level_0_Unnamed: 0_level_1'
        # and the month name is in 'Unnamed: 1_level_0_Unnamed: 1_level_1'.
        column_map = {
            'Unnamed: 0_level_0_Unnamed: 0_level_1': 'Período', # This column contains the year
            'Unnamed: 1_level_0_Unnamed: 1_level_1': 'Mes_Str', # This column contains the month name
            'Exportaciones_Total mensual': 'Exportaciones',
            'Importaciones_Total mensual': 'Importaciones',
            'Saldo_Unnamed: 12_level_1': 'Balanza Comercial'
        }
        df.rename(columns=column_map, inplace=True)

        print("DEBUG: Columnas después de renombrar:")
        print(df.columns.tolist())
        print("DEBUG: df.head() después de renombrar:")
        print(df.head().to_string())

        # Verificar que las columnas necesarias existan después del renombramiento
        # The required columns are now 'Período' (for year), 'Mes_Str' (for month), and the trade values.
        required_cols = ['Período', 'Mes_Str', 'Exportaciones', 'Importaciones', 'Balanza Comercial']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: No se encontraron todas las columnas requeridas ({required_cols}) en el DataFrame después de procesar encabezados multinivel.")
            print(f"Columnas encontradas: {df.columns.tolist()}")
            print("Inspecciona el archivo Excel 'balanmensual.xls' y ajusta el mapeo de columnas y los índices de encabezado.")
            return

        # Separar 'Año' y 'Mes' de la columna 'Período'
        # La columna 'Período' contiene el año, y 'Mes_Str' contiene el nombre del mes.
        
        # Limpiar filas que no son de datos (ej. las que tienen 'Millones de dólares' o NaN en las columnas clave)
        # Esto elimina las primeras filas de metadatos y cualquier fila que no tenga un mes definido.
        # We drop rows where 'Exportaciones' or 'Importaciones' are not numeric, which indicates header rows.
        df = df[pd.to_numeric(df['Exportaciones'], errors='coerce').notna() | pd.to_numeric(df['Importaciones'], errors='coerce').notna()]
        
        # Forward-fill la columna 'Período' (que contiene el año) para propagar el año a todos los meses
        df['Período'] = df['Período'].ffill()

        # Limpiar la columna 'Período' para eliminar caracteres no numéricos (como el asterisco)
        # antes de intentar la conversión a numérico.
        df['Período'] = df['Período'].astype(str).str.replace(r'[^0-9]', '', regex=True)

        # Convert 'Período' to numeric, coercing errors, before dropping NaNs and converting to int.
        # This ensures that any valid year strings are converted correctly.
        df['Período'] = pd.to_numeric(df['Período'], errors='coerce')

        # Eliminar filas donde el 'Período' (año) o 'Mes_Str' (mes) son NaN después del ffill.
        df.dropna(subset=['Período', 'Mes_Str'], inplace=True)

        print("DEBUG: df.head() después de limpiar y ffill 'Período' y 'Mes_Str':")
        print(df.head().to_string())

        # Extract Year from 'Período' (which is now correctly mapped to the year column)
        # Ensure 'Período' is converted to integer after cleaning.
        df['Año'] = df['Período'].astype(int)
        # 'Mes_Str' column already contains the month name, no need to reassign from a problematic source.
        # Ensure it's clean (already handled by mapping and ffill/dropna).

        print("DEBUG: df.head() después de extraer Año y usar Mes_Str:")
        print(df.head().to_string())

        month_map_es = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
            'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
            'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
        }
        
        df['MesNumerico'] = df['Mes_Str'].str.lower().map(month_map_es)
        
        print("DEBUG: df.head() después de mapear MesNumerico:")
        print(df.head().to_string())

        # Convertir 'Año' y 'MesNumerico' a numérico, manejando errores
        df['Año'] = pd.to_numeric(df['Año'], errors='coerce')
        df['MesNumerico'] = pd.to_numeric(df['MesNumerico'], errors='coerce')

        # Eliminar filas donde no se pudo parsear el año o el mes
        df.dropna(subset=['Año', 'MesNumerico'], inplace=True) 
        
        print("DEBUG: df.head() después de dropna en Año y MesNumerico:")
        print(df.head().to_string())

        # Convertir a entero después de eliminar NaNs
        df['Año'] = df['Año'].astype(int)
        df['MesNumerico'] = df['MesNumerico'].astype(int)

        # Filtrar años y meses válidos
        current_year = pd.Timestamp.now().year
        df = df[(df['Año'] >= 1990) & (df['Año'] <= current_year + 1)] # Un margen razonable
        df = df[(df['MesNumerico'] >= 1) & (df['MesNumerico'] <= 12)]

        df["Fecha"] = pd.to_datetime(df["Año"].astype(str) + '-' + df["MesNumerico"].astype(str) + '-01', errors='coerce')

        print("DEBUG: df.head() después de crear Fecha:")
        print(df.head().to_string())

        # Convertir valores a numéricos. Pueden tener separadores de miles o ser texto.
        for col in ["Exportaciones", "Importaciones", "Balanza Comercial"]:
            if col in df.columns:
                # Intentar convertir directamente, luego con reemplazo de puntos si falla
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    # Reemplazar puntos por nada (separador de miles) y comas por puntos (separador decimal)
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')

        df.dropna(subset=["Fecha", "Exportaciones", "Importaciones"], inplace=True)
        df.sort_values(by="Fecha", inplace=True)
        
        print("DEBUG: df.head() final antes de seleccionar columnas para gráfico:")
        print(df.head().to_string())

        # Seleccionar solo las columnas que nos interesan para el gráfico
        df = df[["Fecha", "Exportaciones", "Importaciones", "Balanza Comercial"]].copy()

    except Exception as e:
        print(f"Error al procesar el archivo Excel: {e}")
        print("Esto usualmente ocurre si la estructura del Excel (hoja, encabezados, columnas) no es la esperada.")
        print("Por favor, descarga el archivo Excel manualmente, inspecciónalo y ajusta la lógica de lectura en el script.")
        return

    if df.empty:
        print("No se pudieron extraer datos válidos del archivo Excel o el DataFrame resultante está vacío.")
        return

    # 4. Calcular la balanza comercial (ya debería estar en el DF si se mapeó correctamente)
    # Si 'Balanza Comercial' no se extrajo directamente, calcularla
    if 'Balanza Comercial' not in df.columns:
        df["Balanza Comercial"] = df["Exportaciones"] - df["Importaciones"]

    if df.empty:
        print("No hay datos disponibles para graficar.")
        return

    # 5. Generar un gráfico de líneas con toda la serie histórica
    print("Generando gráfico de la serie histórica completa...")
    plt.figure(figsize=(20, 10)) # Aumentar el tamaño para una serie más larga y más ticks
    plt.plot(df["Fecha"], df["Exportaciones"],
             label="Exportaciones (Millones USD)", marker='', linestyle='-', linewidth=1.5) # Sin marcador para serie larga
    plt.plot(df["Fecha"], df["Importaciones"],
             label="Importaciones (Millones USD)", marker='', linestyle='--', linewidth=1.5) # Sin marcador para serie larga

    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Monto (Millones USD)", fontsize=12)
    plt.title("Balanza Comercial Argentina - Serie Histórica Completa (desde Excel INDEC)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)

    # Ajustar los ticks del eje X para mostrar todos los meses
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # Major ticks for each month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Format major ticks as Year-Month
    # No se usa minor_locator si los major ticks son mensuales

    plt.xticks(rotation=90, ha="center", fontsize=8) # Rotar 90 grados y centrar para mejor legibilidad

    formatter = mticker.FuncFormatter(lambda x, p: format(int(x), ','))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()

    # 6. Imprimir mensaje de confirmación y datos (opcionalmente, solo los primeros/últimos para no saturar)
    print("\nGráfico de la balanza comercial generado exitosamente desde el archivo Excel del INDEC.")
    print(f"URL del archivo utilizado: {EXCEL_URL}")
    print("\nDatos de la serie histórica completa (Millones USD):")
    # Imprimir solo los primeros y últimos 5 registros para no saturar la salida
    print("Primeros 5 registros:")
    print(df.head()[['Fecha', 'Exportaciones', 'Importaciones', 'Balanza Comercial']].to_string(index=False,
        formatters={'Fecha': lambda x: x.strftime('%Y-%m-%d'),
                    'Exportaciones': '{:,.0f}'.format,
                    'Importaciones': '{:,.0f}'.format,
                    'Balanza Comercial': '{:,.0f}'.format
                    }))
    print("\nÚltimos 5 registros:")
    print(df.tail()[['Fecha', 'Exportaciones', 'Importaciones', 'Balanza Comercial']].to_string(index=False,
        formatters={'Fecha': lambda x: x.strftime('%Y-%m-%d'),
                    'Exportaciones': '{:,.0f}'.format,
                    'Importaciones': '{:,.0f}'.format,
                    'Balanza Comercial': '{:,.0f}'.format
                    }))

if __name__ == "__main__":
    obtener_y_graficar_balanza_comercial_excel()
