# Importar las bibliotecas necesarias
import requests  # Para realizar solicitudes HTTP a la API
import matplotlib.pyplot as plt  # Para crear gráficos
import matplotlib.dates as mdates  # Para formatear fechas en los gráficos
from datetime import datetime  # Para trabajar con objetos de fecha y hora
import warnings # Para manejar advertencias de InsecureRequestWarning

# --- Configuración General ---
# URL base de la API de estadísticas monetarias del BCRA
BASE_URL_BCRA = "https://api.bcra.gob.ar/estadisticas/v3.0/monetarias"

# Diccionario de variables predefinidas para facilitar la selección al usuario.
# Este diccionario se basa en el JSON que proporcionaste.
# Contiene el idVariable como clave y la descripción como valor.
VARIABLES_BCRA = {
    1: "Reservas Internacionales del BCRA (en millones de dólares - cifras provisorias sujetas a cambio de valuación)",
    4: "Tipo de Cambio Minorista ($ por USD) Comunicación B 9791 - Promedio vendedor",
    5: "Tipo de Cambio Mayorista ($ por USD) Comunicación A 3500 - Referencia",
    6: "Tasa de Política Monetaria (en % n.a.)",
    7: "BADLAR en pesos de bancos privados (en % n.a.)",
    8: "TM20 en pesos de bancos privados (en % n.a.)",
    9: "Tasas de interés de las operaciones de pase activas para el BCRA, a 1 día de plazo (en % n.a.)",
    10: "Tasas de interés de las operaciones de pase pasivas para el BCRA, a 1 día de plazo (en % n.a.)",
    11: "Tasas de interés por préstamos entre entidades financiera privadas (BAIBAR) (en % n.a.)",
    12: "Tasas de interés por depósitos a 30 días de plazo en entidades financieras (en % n.a.)",
    13: "Tasa de interés de préstamos por adelantos en cuenta corriente",
    14: "Tasa de interés de préstamos personales",
    15: "Base monetaria - Total (en millones de pesos)",
    16: "Circulación monetaria (en millones de pesos)",
    17: "Billetes y monedas en poder del público (en millones de pesos)",
    18: "Efectivo en entidades financieras (en millones de pesos)",
    19: "Depósitos de los bancos en cta. cte. en pesos en el BCRA (en millones de pesos)",
    21: "Depósitos en efectivo en las entidades financieras - Total (en millones de pesos)",
    22: "En cuentas corrientes (neto de utilización FUCO) (en millones de pesos)",
    23: "En Caja de ahorros (en millones de pesos)",
    24: "A plazo (incluye inversiones y excluye CEDROS) (en millones de pesos)",
    25: "M2 privado, promedio móvil de 30 días, variación interanual (en %)",
    26: "Préstamos de las entidades financieras al sector privado (en millones de pesos)",
    27: "Inflación mensual (variación en %)",
    28: "Inflación interanual (variación en % i.a.)",
    29: "Inflación esperada - REM próximos 12 meses - MEDIANA (variación en % i.a)",
    30: "CER (Base 2.2.2002=1)",
    31: "Unidad de Valor Adquisitivo (UVA) (en pesos -con dos decimales-, base 31.3.2016=14.05)",
    32: "Unidad de Vivienda (UVI) (en pesos -con dos decimales-, base 31.3.2016=14.05)",
    34: "Tasa de Política Monetaria (en % e.a.)",
    35: "BADLAR en pesos de bancos privados (en % e.a.)",
    40: "Índice para Contratos de Locación (ICL-Ley 27.551, con dos decimales, base 30.6.20=1)",
    41: "Tasas de interés de las operaciones de pase pasivas para el BCRA, a 1 día de plazo (en % e.a.)",
    42: "Pases pasivos para el BCRA - Saldos (en millones de pesos)",
    43: "Tasa de interés para uso de la Justicia – Comunicado P 14290 | Base 01/04/1991 (en %)",
    44: "TAMAR en pesos de bancos privados (en % n.a.)",
    45: "TAMAR en pesos de bancos privados (en % e.a.)",
    46: "Total de factores de explicación de la variación de la Base Monetaria (en millones de $)",
    47: "Efecto monetario de las compras netas de divisas al sector privado y otros (en millones de $)",
    48: "Efecto monetario de las compras netas de divisas al Tesoro Nacional (en millones de $)",
    49: "Efecto monetario de los adelantos transitorios al Tesoro Nacional (en millones de $)",
    50: "Efecto monetario de las transferencia de utilidades al Tesoro Nacional (en millones de $)",
    51: "Efecto monetario del resto de operaciones con el Tesoro Nacional (en millones de $)",
    52: "Efecto monetario de las operaciones de pases (en millones de $)",
    53: "Efecto monetario de las LELIQ y NOTALIQ (en millones de $)",
    54: "Efecto monetario de los redescuentos y adelantos (en millones de $)",
    55: "Efecto monetario de los intereses, primas y remuneración de cuentas corrientes asociados a op. de pases, LELIQ, NOTALIQ, redescuentos y adel. (en millones de $)",
    56: "Efecto monetario de las LEBAC y NOBAC (en millones de $)",
    57: "Efecto monetario del rescate de cuasimonedas (en millones de $)",
    58: "Efecto monetario de las operaciones con Letras Fiscales de Liquidez (en millones de $)",
    59: "Otras operaciones que explican la variación de la base monetaria (en millones de $)",
    60: "Variación diaria de billetes y monedas en poder del público (en millones de $)",
    61: "Variación diaria de billetes y monedas en entidades financieras (en millones de $)",
    62: "Variación diaria de cheques cancelatorios (en millones de $)",
    63: "Variación diaria de cuentas corrientes en pesos en el BCRA (en millones de $)",
    64: "Variación diaria de la base monetaria (en millones de $)",
    65: "Variación diaria de cuasimonedas (en millones de $)",
    66: "Variación diaria de la base monetaria más variación diaria de cuasimonedas (en millones de $)",
    67: "Saldo de billetes y monedas en poder del público (en millones de $)",
    68: "Saldo de billetes y monedas en entidades financieras (en millones de $)",
    69: "Saldo de cheques cancelatorios (en millones de $)",
    70: "Saldo de cuentas corrientes en pesos en el BCRA (en millones de $)",
    71: "Saldo de base monetaria (en millones de $)",
    72: "Saldo de cuasimonedas (en millones de $)",
    73: "Saldo de base monetaria más cuasimonedas (en millones de $)",
    74: "Saldo de reservas internacionales (excluidas asignaciones DEG 2009, en millones de USD)",
    75: "Saldo de oro, divisas, colocaciones a plazo y otros activos de reserva (en millones de USD)",
    76: "Saldo de divisas-pase pasivo en dólares con el exterior concertados en 2016 (en millones de USD)",
    77: "Total de variación diaria de las reservas internacionales (en millones de USD)",
    78: "Variación diaria de reservas internacionales por compra de divisas (en millones de USD)",
    79: "Variación diaria de reservas internacionales por operaciones con organismos internacionales (en millones de USD)",
    80: "Variación diaria de reservas internacionales por otras operaciones del sector público (en millones de USD)",
    81: "Variación diaria de reservas internacionales por efectivo mínimo (en millones de USD)",
    82: "Variación diaria de reservas internacionales por otras operaciones no incluidas en otros rubros (en millones de USD)", # Corregido typo 'outras' a 'otras'
    83: "Saldo de Asignaciones de DEGs del año 2009 (en millones de USD)",
    84: "Tipo de cambio peso / dólar estadounidense de valuación contable",
    85: "Saldo de depósitos en pesos en cuentas corrientes de los sectores público y privado no financieros (en millones de $)",
    86: "Saldo de depósitos en pesos en cajas de ahorro de los sectores público y privado no financieros (en millones de $)",
    87: "Saldo de depósitos en pesos a plazo no ajustables por CER/UVAs de los sectores público y privado no financieros (en millones de $)",
    88: "Saldo de depósitos en pesos a plazo ajustables por CER/UVAs de los sectores público y privado no financieros (en millones de $)",
    89: "Saldo de otros depósitos en pesos de los sectores público y privado no financieros (en millones de $)",
    90: "Saldo de CEDROS con CER de los sectores público y privado no financieros (en millones de $)",
    91: "Saldo de los depósitos en pesos de los sectores público y privados no financieros más CEDROS (en millones de $)",
    92: "Saldo de BODEN de los sectores público y privado no financieros (en millones de $)",
    93: "Saldo de los depósitos en pesos de los sectores público y privados no financieros más CEDRO más BODEN (en millones de $)",
    94: "Saldo de depósitos en pesos cuentas corrientes del sector privado no financiero (en millones de $)",
    95: "Saldo de depósitos en pesos en cajas de ahorro del sector privado no financiero (en millones de $)",
    96: "Saldo de depósitos en pesos a plazo no ajustables por CER/UVAs del sector privado no financiero (en millones de $)",
    97: "Saldo de depósitos en pesos a plazo ajustables por CER/UVAs del sector privado no financiero (en millones de $)",
    98: "Saldo de otros depósitos en pesos del sector privado no financiero (en millones de $)",
    99: "Saldo de CEDROS con CER del sector privado no financiero (en millones de $)",
    100: "Saldo de los depósitos en pesos del sector privado no financiero más CEDROS (en millones de $)",
    101: "Saldo de BODEN del sector privado no financiero (en millones de $)",
    102: "Saldo de los depósitos en pesos del sector privado no financiero más CEDRO más BODEN (en millones de $)",
    103: "Saldo de depósitos en dólares de los sectores público y privado no financieros, expresados en pesos (en millones de $)",
    104: "Saldo de depósitos en dólares del sector privado no financiero, expresados en pesos (en millones de $)",
    105: "Saldo de depósitos en pesos y en dólares de los sectores público y privado no financieros, expresados en pesos (en millones de $)",
    106: "Saldo de depósitos en pesos y dólares del sector privado no financiero, expresados en pesos (en millones de $)",
    107: "Saldo de depósitos en dólares de los sectores público y privado no financieros, expresados en dólares (en millones de USD)",
    108: "Saldo de depósitos en dólares del sector privado no financiero, expresados en dólares (en millones de USD)",
    109: "Saldo del agregado monetario M2 (billetes y monedas en poder del público y depósitos en cuenta corriente y en caja de ahorro en pesos correspondientes al sector privado y al sector público, en millones de $)",
    110: "Saldo de préstamos otorgados al sector privado mediante adelantos en cuenta corriente en pesos (en millones de $)",
    111: "Saldo de préstamos otorgados al sector privado mediante documentos en pesos (en millones de $)",
    112: "Saldo de préstamos hipotecarios en pesos otorgados al sector privado (en millones de $)",
    113: "Saldo de préstamos prendarios en pesos otorgados al sector privado (en millones de $)",
    114: "Saldo de préstamos personales en pesos (en millones de $)",
    115: "Saldo de préstamos en pesos mediante tarjetas de crédito otorgados al sector privado (en millones de $)",
    116: "Saldo de otros préstamos en pesos otorgados al sector privado (en millones de $)",
    117: "Saldo total de préstamos al sector privado en pesos (en millones de $)",
    118: "Saldo de préstamos otorgados al sector privado mediante adelantos en cuenta corriente en dólares (en millones de USD)",
    119: "Saldo de préstamos otorgados al sector privado mediante documentos en dólares (en millones de USD)",
    120: "Saldo de préstamos hipotecarios en dólares otorgados al sector privado (en millones de USD)",
    121: "Saldo de préstamos prendarios en dólares otorgados al sector privado (en millones de USD)",
    122: "Saldo de préstamos personales en dólares (en millones de USD)",
    123: "Saldo de préstamos en dólares mediante tarjetas de crédito otorgados al sector privado(en millones de USD)",
    124: "Saldo de otros préstamos en dólares otorgados al sector privado (en millones de USD)",
    125: "Saldo total de préstamos otorgados al sector privado en dólares (en millones de USD)",
    126: "Saldo total de préstamos otorgados al sector privado en dólares, expresado en pesos (en millones de $)",
    127: "Saldo total de préstamos otorgados del sector privado en pesos y moneda extranjera, expresado en pesos (en millones de $)",
    128: "Tasa de interés de depósitos a plazo fijo en pesos, de 30-44 días , total de operaciones,TNA (en %)",
    129: "Tasa de interés de depósitos a plazo fijo en pesos, de 30-44 días, hasta $100.000, TNA (en %)",
    130: "Tasa de interés de depósitos a plazo fijo en pesos, de 30-44 días, hasta $100.000, TEA (en %)",
    131: "Tasa de interés de depósitos a plazo fijo en pesos, de 30-44 días, de más de $1.000.000, TNA (en %)",
    132: "Tasa de interés de depósitos a plazo fijo en dólares, de 30-44 días, total de operaciones, TNA (en %)",
    133: "Tasa de interés de depósitos a plazo fijo en dólares, de 30-44 días, hasta $100.000, TNA (en %)",
    134: "Tasa de interés de depósitos a plazo fijo en dólares, de 30-44 días, de mas de USD1.000.000, TNA (en %)",
    135: "TAMAR total bancos, TNA (en %)",
    136: "TAMAR de bancos privados,TNA (en %)",
    137: "TAMAR de bancos privados,TEA (en %)",
    138: "BADLAR total bancos, TNA (en %)",
    139: "BADLAR de bancos privados,TNA (en %)",
    140: "BADLAR de bancos privados,TEA (en %)",
    141: "TM20 total bancos, TNA (en %)",
    142: "TM20 de bancos privados, TNA (en %)",
    143: "TM20 de bancos privados, TEA (en %)",
    144: "Tasa de interés de préstamos personales en pesos, TNA (en %)",
    145: "Tasa de interés por adelantos en cuenta corriente en pesos, con acuerdo de 1 a 7 días y de 10 millones o más, a empresas del sector privado, TNA (en %)",
    146: "Tasa de interés por operaciones de préstamos entre entidades financieras locales privadas (BAIBAR, TNA, en %)",
    147: "Monto de operaciones de préstamos entre entidades financieras locales privados (BAIBAR, en millones de $)",
    148: "Tasa de interés por operaciones de préstamos entre entidades financieras locales, TNA (en %)", # Corregido typo 'interes' a 'interés'
    149: "Monto de operaciones de préstamos entre entidades financieras locales (en millones de $)",
    150: "Tasa de interés por operaciones de pases entre terceros a 1 día, TNA (en %)", # Corregido typo 'interes' a 'interés'
    151: "Monto de operaciones de pases entre terceros (en millones de $)",
    152: "Saldo total de pases pasivos para el BCRA (incluye pases pasivos con FCI, en millones de $)",
    153: "Saldo de pases pasivos del BCRA con fondos comunes de inversión (en millones de $)",
    154: "Saldo de pases activos para el BCRA (en millones de $)",
    155: "Saldo de LELIQ y NOTALIQ (en millones de $)",
    156: "Saldo de LEBAC y NOBAC en Pesos, LEGAR y LEMIN (en millones de $)",
    157: "Saldo de LEBAC y NOBAC en Pesos de Entidades Financieras (en millones de $)",
    158: "Saldo de LEBAC en dólares, LEDIV y BOPREAL (en millones de USD)",
    159: "Saldo de NOCOM (en millones de $)",
    160: "Tasas de interés de política monetaria, TNA (en %)",
    161: "Tasas de interés de política monetaria, TEA (en %)",
    162: "Tasas de interés del BCRA para pases pasivos en pesos a 1 día, TNA (en %)",
    163: "Tasas de interés del BCRA para pases pasivos en pesos a 7 días, TNA (en %)",
    164: "Tasas de interés del BCRA para pases activos en pesos a 1 días, TNA (en %)",
    165: "Tasas de interés del BCRA para pases activos en pesos a 7 días, TNA (en %)",
    166: "Tasas de interés de LEBAC en Pesos / LELIQ de 1 mes, TNA (en %)",
    167: "Tasas de interés de LEBAC en Pesos de 2 meses, TNA (en %)",
    168: "Tasas de interés de LEBAC en Pesos de 3 meses, TNA (en %)",
    169: "Tasas de interés de LEBAC en Pesos de 4 meses, TNA (en %)",
    170: "Tasas de interés de LEBAC en Pesos de 5 meses, TNA (en %)",
    171: "Tasas de interés de LEBAC en Pesos / LELIQ a 6 meses, TNA (en %)",
    172: "Tasas de interés de LEBAC en Pesos de 7 meses, TNA (en %)",
    173: "Tasas de interés de LEBAC en Pesos de 8 meses, TNA (en %)",
    174: "Tasas de interés de LEBAC en Pesos de 9 meses, TNA (en %)",
    175: "Tasas de interés de LEBAC en Pesos de 10 meses, TNA (en %)",
    176: "Tasas de interés de LEBAC en Pesos de 11 meses, TNA (en %)",
    177: "Tasas de interés de LEBAC en Pesos de 12 meses, TNA (en %)",
    178: "Tasas de interés de LEBAC en Pesos de 18 meses, TNA (en %)",
    179: "Tasas de interés de LEBAC en Pesos de 24 meses, TNA (en %)",
    180: "Tasas de interés de LEBAC en pesos ajustables por CER de 6 meses, TNA (en %)",
    181: "Tasas de interés de LEBAC en pesos ajustables por CER de 12 meses, TNA (en %)",
    182: "Tasas de interés de LEBAC en pesos ajustables por CER de 18 meses, TNA (en %)",
    183: "Tasas de interés de LEBAC en pesos ajustables por CER de 24 meses, TNA (en %)",
    184: "Tasas de interés de LEBAC en dólares, con liquidación en pesos, de 1 mes, TNA (en %)",
    185: "Tasas de interés de LEBAC en dólares, con liquidación en pesos, de 6 meses, TNA (en %)",
    186: "Tasas de interés de LEBAC en dólares, con liquidación en pesos, de 12 meses, TNA (en %)",
    187: "Tasas de interés de LEBAC en dólares, con liquidación en dólares, de 1 mes, TNA (en %)",
    188: "Tasas de interés de LEBAC en dólares, con liquidación en dólares, de 3 meses, TNA (en %)",
    189: "Tasas de interés de LEBAC en dólares, con liquidación en dólares, de 6 meses, TNA (en %)",
    190: "Tasas de interés de LEBAC en dólares, con liquidación en dólares, de 12 meses, TNA (en %)",
    191: "Margen sobre BADLAR Bancos Privados de NOBAC de 9 meses (en %)",
    192: "Margen sobre Bancos Privados de NOBAC de 12 meses (en %)",
    193: "Margen sobre BADLAR Total de NOBAC de 2 Años (en %)",
    194: "Margen sobre BADLAR Bancos Privados de NOBAC de 2 Años (en %)",
    195: "Margen sobre Tasa de Política Monetaria de NOTALIQ en Pesos de 190 dias (en %)", # Corregido typo 'Politica' a 'Política'
    196: "Saldo de Letras Fiscales de Liquidez en cartera de entidades financieras, en valor técnico (en millones de $)",
    197: "Saldo de M2 Transaccional del Sector Privado (expresado en millones de Pesos)"
}

# --- Funciones ---

def listar_variables_disponibles():
    """
    Muestra al usuario las variables disponibles para consultar,
    utilizando el diccionario VARIABLES_BCRA.
    """
    print("Variables disponibles para consultar en la API del BCRA:")
    print("-------------------------------------------------------")
    for var_id, desc in VARIABLES_BCRA.items():
        print(f"ID: {var_id} - Descripción: {desc}")
    print("-------------------------------------------------------")

def validar_fecha_iso(fecha_str):
    """
    Valida si una cadena de texto de fecha está en formato ISO 8601 (YYYY-MM-DD).
    Retorna True si es válida, False en caso contrario.
    """
    try:
        datetime.strptime(fecha_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def fetch_bcra_data(variable_id, fecha_desde=None, fecha_hasta=None, limit=1000):
    """
    Obtiene datos de una estadística específica del BCRA para un ID de variable dado.

    Args:
        variable_id (int): El ID de la variable a consultar (obtenido de VARIABLES_BCRA).
        fecha_desde (str, optional): Fecha de inicio del rango en formato 'YYYY-MM-DD'.
                                     Defaults to None (sin filtro de inicio).
        fecha_hasta (str, optional): Fecha de fin del rango en formato 'YYYY-MM-DD'.
                                     Defaults to None (sin filtro de fin).
        limit (int, optional): Cantidad máxima de registros a retornar por la API.
                               Valor máximo permitido por la API es 3000. Defaults to 1000.

    Returns:
        list: Una lista de diccionarios, donde cada diccionario contiene 'fecha' (objeto date)
              y 'valor' (float). Retorna None si ocurre un error o no se encuentran datos.
    """
    # Construir la URL completa para el endpoint específico de la variable
    url = f"{BASE_URL_BCRA}/{variable_id}"

    # Preparar los parámetros para la solicitud GET
    params = {}
    if fecha_desde:
        params['desde'] = fecha_desde
    if fecha_hasta:
        params['hasta'] = fecha_hasta
    
    params['limit'] = min(limit, 3000)

    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
               'referer':'https://www.google.com/'}

    
    ssl_verify = False # Por defecto, la verificación SSL está activada (True)
    # ssl_verify = False # DESCOMENTA ESTA LÍNEA PARA DESHABILITAR LA VERIFICACIÓN SSL (MENOS SEGURO)

    if not ssl_verify:
        # Suprimir solo las advertencias de solicitud insegura si se deshabilita la verificación SSL
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')


    print(f"\nRealizando consulta a la API del BCRA:")
    print(f"URL: {url}")
    print(f"Parámetros: {params}")
    print(f"Headers: {headers}")
    print(f"Verificación SSL: {'Activada' if ssl_verify else 'DESACTIVADA (RIESGO DE SEGURIDAD)'}")


    try:
        # Realizar la solicitud GET a la API, incluyendo los headers y la opción de verificación SSL
        response = requests.get(url, params=params, headers=headers, timeout=20, verify=ssl_verify)
        response.raise_for_status()

        data = response.json()

        if data.get('status') == 200:
            resultados_procesados = []
            for item in data.get('results', []):
                try:
                    fecha_obj = datetime.strptime(item['fecha'], '%Y-%m-%d').date()
                    valor_float = float(item['valor'])
                    resultados_procesados.append({'fecha': fecha_obj, 'valor': valor_float})
                except (ValueError, TypeError) as e:
                    print(f"Advertencia: No se pudo procesar el registro: {item}. Error: {e}")
            
            resultados_procesados.sort(key=lambda x: x['fecha'])
            return resultados_procesados
        else:
            print(f"Error en la respuesta de la API: Status {data.get('status')}")
            error_msg = data.get('developerMessage') or data.get('userMessage') or data.get('errorMessage') or "No hay mensaje de error detallado."
            print(f"Mensaje de la API: {error_msg}")
            return None

    except requests.exceptions.SSLError as ssl_err:
        print(f"Error de SSL al contactar la API: {ssl_err}")
        print("Esto puede deberse a certificados desactualizados en tu sistema.")
        print("Intenta actualizar 'certifi': pip install --upgrade certifi")
        print("Si eso no funciona, como último recurso (y entendiendo los riesgos de seguridad),")
        print("puedes desactivar la verificación SSL en el script descomentando la línea 'ssl_verify = False'.")
    except requests.exceptions.HTTPError as http_err:
        print(f"Error HTTP al contactar la API: {http_err}")
        if response is not None:
            print(f"Contenido de la respuesta (si disponible): {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Error de Conexión con la API: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Error de Timeout: La solicitud a la API tardó demasiado en responder: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Error general en la solicitud a la API: {req_err}")
    except ValueError as json_err:
        print(f"Error al decodificar la respuesta JSON de la API: {json_err}")
    return None

def get_minorista_exchange_rate(fecha_desde=None, fecha_hasta=None):
    """
    Obtiene el tipo de cambio minorista (ID 4) para un rango de fechas.
    """
    print("\nObteniendo datos del Tipo de Cambio Minorista para la conversión...")
    return fetch_bcra_data(4, fecha_desde, fecha_hasta)

def get_cer_data(fecha_desde=None, fecha_hasta=None):
    """
    Obtiene el Coeficiente de Estabilización de Referencia (CER, ID 30) para un rango de fechas.
    """
    print("\nObteniendo datos del CER para la conversión...")
    return fetch_bcra_data(30, fecha_desde, fecha_hasta)

def adjust_data_by_index(data_serie, index_data, index_name="Índice"):
    """
    Ajusta los valores de una serie de datos utilizando un índice proporcionado.

    Args:
        data_serie (list): Lista de diccionarios con 'fecha' y 'valor' de la serie principal.
        index_data (list): Lista de diccionarios con 'fecha' y 'valor' del índice (tipo de cambio, CER, etc.).
        index_name (str, optional): Nombre del índice para mensajes de advertencia. Defaults to "Índice".

    Returns:
        list: Nueva lista de diccionarios con los valores ajustados.
    """
    if not index_data:
        print(f"Advertencia: No se pudo obtener el {index_name}. No se realizará el ajuste.")
        return data_serie

    # Crear un diccionario para una búsqueda rápida del índice por fecha
    index_map = {item['fecha']: item['valor'] for item in index_data}

    adjusted_data = []
    for item in data_serie:
        fecha = item['fecha']
        valor = item['valor']
        
        index_value = index_map.get(fecha)
        
        if index_value is not None and index_value != 0:
            adjusted_value = valor / index_value
            adjusted_data.append({'fecha': fecha, 'valor': adjusted_value})
        else:
            print(f"Advertencia: No se encontró {index_name} para la fecha {fecha} o el valor es cero. Se omite el punto.")
    return adjusted_data

def plot_single_series(datos_serie, titulo_grafico, nombre_eje_y="Valor"):
    """
    Crea y muestra un gráfico de línea con los datos de una serie temporal proporcionada.
    """
    if not datos_serie:
        print("No hay datos para graficar.")
        return

    fechas = [item['fecha'] for item in datos_serie]
    valores = [item['valor'] for item in datos_serie]

    plt.figure(figsize=(14, 7))
    plt.plot(fechas, valores, marker='o', linestyle='-', markersize=4, color='b', label=titulo_grafico)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=15))
    plt.xticks(rotation=45, ha="right")

    plt.xlabel("Fecha")
    plt.ylabel(nombre_eje_y)
    plt.title(titulo_grafico, fontsize=16)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_two_series(data_serie_1, desc_1, y_label_1, data_serie_2, desc_2, y_label_2, titulo_grafico):
    """
    Crea y muestra un gráfico de línea con dos series temporales, utilizando dos ejes Y.
    """
    if not data_serie_1 or not data_serie_2:
        print("No hay datos suficientes para graficar dos series.")
        return

    # Asegurar que ambas series estén ordenadas por fecha
    data_serie_1.sort(key=lambda x: x['fecha'])
    data_serie_2.sort(key=lambda x: x['fecha'])

    # Encontrar fechas comunes para alinear los datos
    fechas_1 = {item['fecha'] for item in data_serie_1}
    fechas_2 = {item['fecha'] for item in data_serie_2}
    fechas_comunes = sorted(list(fechas_1.intersection(fechas_2)))

    if not fechas_comunes:
        print("No hay fechas comunes entre las dos series para graficar.")
        return

    # Filtrar y alinear los datos a las fechas comunes
    aligned_data_1 = {item['fecha']: item['valor'] for item in data_serie_1 if item['fecha'] in fechas_comunes}
    aligned_data_2 = {item['fecha']: item['valor'] for item in data_serie_2 if item['fecha'] in fechas_comunes}

    fechas_plot = [f for f in fechas_comunes]
    valores_1 = [aligned_data_1[f] for f in fechas_comunes]
    valores_2 = [aligned_data_2[f] for f in fechas_comunes]

    fig, ax1 = plt.subplots(figsize=(14, 7))

    color1 = 'tab:blue'
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel(y_label_1, color=color1)
    ax1.plot(fechas_plot, valores_1, color=color1, marker='o', linestyle='-', markersize=4, label=desc_1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=15))
    plt.xticks(rotation=45, ha="right")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()  # Instancia un segundo eje que comparte el mismo eje x
    color2 = 'tab:red'
    ax2.set_ylabel(y_label_2, color=color2)
    ax2.plot(fechas_plot, valores_2, color=color2, marker='o', linestyle='-', markersize=4, label=desc_2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.suptitle(titulo_grafico, fontsize=16)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    fig.tight_layout()
    plt.show()


# --- Script Principal (punto de entrada de la ejecución) ---
if __name__ == "__main__":
    listar_variables_disponibles()

    while True:
        try:
            plot_choice = input("¿Qué desea graficar? (1. Una sola variable / 2. Dos variables a la vez): ")
            if plot_choice in ['1', '2']:
                break
            else:
                print("Opción no válida. Por favor, ingrese '1' o '2'.")
        except ValueError:
            print("Entrada no válida. Por favor, ingrese un número.")

    fecha_desde_input = input("Ingrese la fecha de inicio (formato YYYY-MM-DD, opcional, presione Enter para omitir): ")
    if fecha_desde_input and not validar_fecha_iso(fecha_desde_input):
        print("Formato de fecha de inicio incorrecto. Se omitirá el filtro de fecha de inicio.")
        fecha_desde_input = None

    fecha_hasta_input = input("Ingrese la fecha de fin (formato YYYY-MM-DD, opcional, presione Enter para omitir): ")
    if fecha_hasta_input and not validar_fecha_iso(fecha_hasta_input):
        print("Formato de fecha de fin incorrecto. Se omitirá el filtro de fecha de fin.")
        fecha_hasta_input = None
    
    limite_registros = 3000

    if plot_choice == '1':
        # --- Flujo para una sola variable ---
        while True:
            try:
                id_variable_seleccionado = int(input("Ingrese el ID de la variable que desea consultar: "))
                if id_variable_seleccionado in VARIABLES_BCRA:
                    break
                else:
                    print("ID de variable no válido. Por favor, elija un ID de la lista.")
            except ValueError:
                print("Entrada no válida. Por favor, ingrese un número entero para el ID.")

        descripcion_variable = VARIABLES_BCRA.get(id_variable_seleccionado, f"Estadística BCRA con ID {id_variable_seleccionado}")
        datos_api = fetch_bcra_data(id_variable_seleccionado, fecha_desde_input, fecha_hasta_input, limit=limite_registros)

        if datos_api:
            print(f"\nSe obtuvieron {len(datos_api)} registros para la variable '{descripcion_variable}'.")
            
            is_in_pesos = "pesos" in descripcion_variable.lower() and "usd" not in descripcion_variable.lower()

            if is_in_pesos:
                while True:
                    print("\nOpciones de ajuste:")
                    print("1. Ajustar a Dólar Constante (Tipo de Cambio Minorista)")
                    print("2. Ajustar a Pesos Constantes (CER)")
                    print("3. No ajustar")
                    opcion_ajuste = input("Ingrese el número de la opción deseada: ")

                    if opcion_ajuste == '1':
                        index_data = get_minorista_exchange_rate(fecha_desde_input, fecha_hasta_input)
                        index_name = "Tipo de Cambio Minorista"
                        new_y_label = "Valor (USD Constante)"
                        new_title_suffix = "(en Dólares Constantes)"
                        break
                    elif opcion_ajuste == '2':
                        index_data = get_cer_data(fecha_desde_input, fecha_hasta_input)
                        index_name = "CER"
                        new_y_label = "Valor (Pesos Constantes - CER)"
                        new_title_suffix = "(en Pesos Constantes - CER)"
                        break
                    elif opcion_ajuste == '3':
                        index_data = None # No adjustment
                        index_name = None
                        new_y_label = f"Valor ({descripcion_variable.split('(')[-1].split(')')[0]})" if '(' in descripcion_variable else "Valor"
                        new_title_suffix = ""
                        break
                    else:
                        print("Opción no válida. Por favor, ingrese 1, 2 o 3.")
                
                if index_data:
                    datos_api_ajustados = adjust_data_by_index(datos_api, index_data, index_name)
                    if datos_api_ajustados:
                        print(f"Se ajustaron {len(datos_api_ajustados)} registros.")
                        plot_single_series(datos_api_ajustados, 
                                  titulo_grafico=f"{descripcion_variable} {new_title_suffix}", 
                                  nombre_eje_y=new_y_label)
                    else:
                        print("No se pudieron ajustar los datos. Graficando valores originales.")
                        plot_single_series(datos_api, titulo_grafico=descripcion_variable, nombre_eje_y=f"Valor ({descripcion_variable.split('(')[-1].split(')')[0]})" if '(' in descripcion_variable else "Valor")
                else: # No adjustment chosen or index data not found
                    plot_single_series(datos_api, titulo_grafico=descripcion_variable, nombre_eje_y=f"Valor ({descripcion_variable.split('(')[-1].split(')')[0]})" if '(' in descripcion_variable else "Valor")
            else:
                print("La variable seleccionada no parece estar en pesos. No se ofrecerá la opción de ajuste a dólar constante ni CER.")
                plot_single_series(datos_api, titulo_grafico=descripcion_variable, nombre_eje_y=f"Valor ({descripcion_variable.split('(')[-1].split(')')[0]})" if '(' in descripcion_variable else "Valor")
        else:
            print(f"No se pudieron obtener o procesar los datos para la variable con ID {id_variable_seleccionado}.")

    elif plot_choice == '2':
        # --- Flujo para dos variables ---
        while True:
            try:
                id_variable_1 = int(input("Ingrese el ID de la PRIMERA variable a consultar: "))
                if id_variable_1 in VARIABLES_BCRA:
                    break
                else:
                    print("ID de variable no válido. Por favor, elija un ID de la lista.")
            except ValueError:
                print("Entrada no válida. Por favor, ingrese un número entero para el ID.")
        
        while True:
            try:
                id_variable_2 = int(input("Ingrese el ID de la SEGUNDA variable a consultar: "))
                if id_variable_2 in VARIABLES_BCRA:
                    break
                else:
                    print("ID de variable no válido. Por favor, elija un ID de la lista.")
            except ValueError:
                print("Entrada no válida. Por favor, ingrese un número entero para el ID.")

        desc_variable_1 = VARIABLES_BCRA.get(id_variable_1, f"Estadística BCRA con ID {id_variable_1}")
        desc_variable_2 = VARIABLES_BCRA.get(id_variable_2, f"Estadística BCRA con ID {id_variable_2}")

        datos_api_1 = fetch_bcra_data(id_variable_1, fecha_desde_input, fecha_hasta_input, limit=limite_registros)
        datos_api_2 = fetch_bcra_data(id_variable_2, fecha_desde_input, fecha_hasta_input, limit=limite_registros)

        if datos_api_1 and datos_api_2:
            print(f"\nSe obtuvieron {len(datos_api_1)} registros para '{desc_variable_1}' y {len(datos_api_2)} para '{desc_variable_2}'.")

            is_var1_in_pesos = "pesos" in desc_variable_1.lower() and "usd" not in desc_variable_1.lower()
            is_var2_in_pesos = "pesos" in desc_variable_2.lower() and "usd" not in desc_variable_2.lower()

            current_y_label_1 = f"Valor ({desc_variable_1.split('(')[-1].split(')')[0]})" if '(' in desc_variable_1 else "Valor"
            current_y_label_2 = f"Valor ({desc_variable_2.split('(')[-1].split(')')[0]})" if '(' in desc_variable_2 else "Valor"
            
            final_title_suffix = ""

            if is_var1_in_pesos and is_var2_in_pesos:
                while True:
                    print("\nOpciones de ajuste unificado para ambas variables (ambas están en pesos):")
                    print("1. Ajustar a Dólar Constante (Tipo de Cambio Minorista)")
                    print("2. Ajustar a Pesos Constantes (CER)")
                    print("3. No ajustar")
                    opcion_ajuste_unificado = input("Ingrese el número de la opción deseada: ")

                    if opcion_ajuste_unificado == '1':
                        index_data = get_minorista_exchange_rate(fecha_desde_input, fecha_hasta_input)
                        index_name = "Tipo de Cambio Minorista"
                        if index_data:
                            datos_api_1 = adjust_data_by_index(datos_api_1, index_data, index_name)
                            datos_api_2 = adjust_data_by_index(datos_api_2, index_data, index_name)
                            current_y_label_1 = "Valor (USD Constante)"
                            current_y_label_2 = "Valor (USD Constante)"
                            final_title_suffix = "(en Dólares Constantes)"
                        else:
                            print("No se pudo obtener el tipo de cambio minorista. No se realizará el ajuste unificado.")
                        break
                    elif opcion_ajuste_unificado == '2':
                        index_data = get_cer_data(fecha_desde_input, fecha_hasta_input)
                        index_name = "CER"
                        if index_data:
                            datos_api_1 = adjust_data_by_index(datos_api_1, index_data, index_name)
                            datos_api_2 = adjust_data_by_index(datos_api_2, index_data, index_name)
                            current_y_label_1 = "Valor (Pesos Constantes - CER)"
                            current_y_label_2 = "Valor (Pesos Constantes - CER)"
                            final_title_suffix = "(en Pesos Constantes - CER)"
                        else:
                            print("No se pudo obtener el CER. No se realizará el ajuste unificado.")
                        break
                    elif opcion_ajuste_unificado == '3':
                        break # No adjustment
                    else:
                        print("Opción no válida. Por favor, ingrese 1, 2 o 3.")
            else:
                print("Al menos una de las variables no parece estar en pesos. No se ofrecerá la opción de ajuste unificado.")
            
            plot_two_series(datos_api_1, desc_variable_1, current_y_label_1, 
                            datos_api_2, desc_variable_2, current_y_label_2, 
                            titulo_grafico=f"{desc_variable_1} vs {desc_variable_2} {final_title_suffix}")
        else:
            print(f"No se pudieron obtener o procesar los datos para una o ambas variables.")
