import requests
from bs4 import BeautifulSoup
import re, json, io

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st
import yfinance as yf
from fpdf import FPDF
from PIL import Image

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'DNT': '1',  # Do Not Track Request Header
    'Connection': 'close'
}

def sec_agent(ticker):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        max_tokens=None,
        max_retries=6,
        stop=None
    )

    stock = yf.Ticker(ticker)
    data = stock.sec_filings
    logo_url = f"https://logo.clearbit.com/{stock.info.get('website', '').replace('http://', '').replace('https://', '').split('/')[0]}"

    # Find the last 10-K or 10-Q filing and get its edgarUrl
    latest_filing = None
    for item in data:
        if isinstance(item, dict) and "type" in item and item["type"] in ["10-K", "10-Q"]:
            if item["type"] == "10-K":
                latest_filing = item['exhibits'].get("10-K")
                break
            elif item["type"] == "10-Q":
                latest_filing = item['exhibits'].get("10-Q")
                break

    if latest_filing is None:
        st.error("No 10-K or 10-Q filing found.")
        st.json(data)
        return
    
    response = requests.get(latest_filing, headers=headers)

    soup = BeautifulSoup(response.content, "html.parser")

    text_content = soup.get_text()
    # Create a prompt template for the analysis
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
            """Eres un experto inversor y analista financiero especializado en reportes 10-K y 10-Q. 
              Tu tarea es analizar el reporte proporcionado y devolver un análisis completo en formato JSON. 
              Cada sección debe ser breve y muy específica, basada en los datos del reporte. El formato es el siguiente (no uses markdown ni texto adicional):
              "filing":
                "type": "Tipo de reporte.",
                "date": "Fecha del reporte."
              ,
              "overview": 
                "description": "Breve descripción de la actividad principal de la empresa.",
                "industry_position": "Posición en la industria, ventajas competitivas o desventajas. De dónde proviene el mayor ingreso."
              ,
              "financial_performance": 
                "revenue": "Ingresos reportados y su variación respecto al periodo anterior.",
                "profit_margins": "Cambios en márgenes de ganancia y rentabilidad.",
                "key_metrics": "Indicadores financieros clave (ej. EPS, flujo de caja, P/B, P/FCF, Debt/Eq, etc). Analizarlos de acuerdo al sector."
              ,
              "risks": 
                "highlighted_risks": "Principales riesgos mencionados en el reporte.",
                "potential_impact": "Impacto potencial de estos riesgos en el negocio."
              ,
              "strategies_future": 
                "short_term": "Estrategias a corto plazo.",
                "long_term": "Planes estratégicos a largo plazo.",
                "innovations": "Nuevas tecnologías o productos mencionados."
              ,
              "projections": 
                "growth": "Proyecciones de crecimiento o contracción.",
                "key_drivers": "Factores clave que impulsan estas proyecciones."
              ,
              "management": 
                "evaluation": "Análisis de la gestión actual.",
                "organizational_changes": "Cambios significativos en la estructura."
              ,
              "corporate_actions": 
                "mergers_acquisitions": "Fusiones o adquisiciones recientes.",
                "new_investments": "Inversiones en mercados o tecnologías nuevas."
              ,
              "conclusion": 
                "investment_recommendation": "¿Es una buena inversión? Responde con 'Sí' o 'No'.",
                "reasoning": "Argumentos clave para la recomendación.",
                "opportunities": "Oportunidades destacadas.",
                "risks": "Riesgos principales."
            """
            ),
            ("human", f"Analiza este reporte: {text_content} y haz un breve resumen. Agrega una pequeña sintesis con argumentos si es buena inversión o no. No incluyas JSON markdown."),
            ]
    )
    prompt = template.invoke({"text_content": text_content})
    result = llm.invoke(prompt)

    try:
        cleaned_result = re.sub(r"\s*```json\s*", '', result.content).strip()
        cleaned_result = re.sub(r"\```", '"', cleaned_result)  # Replace single quotes with double quotes
        cleaned_result = re.sub(r'\s+', ' ', cleaned_result)  # Replace multiple spaces with a single space
        json_match = re.search(r'(\{.*\})', cleaned_result)
    except Exception as e:
        st.error(f"Error processing JSON: {e}")
        return None

    return json_match.group(1), logo_url

# Función para formatear cada sección
def display_section(title, content):
    st.markdown(f"### {title}")
    for key, value in content.items():
        # Resaltar las claves y formatear valores
        st.markdown(f"**{key.replace('_', ' ').capitalize()}:** {value}")

# Crear el contenido del PDF
def create_pdf(data, ticker, logo_url):
  pdf = FPDF()
  pdf.set_auto_page_break(auto=True, margin=15)
  pdf.add_page()
  pdf.set_font("Courier", size=12)

  # Add company logo and ticker
  logo_response = requests.get(logo_url)
  logo_image = Image.open(io.BytesIO(logo_response.content))
  logo_image_path = "logo.png"
  logo_image.save(logo_image_path)
  pdf.image(logo_image_path, x=10, y=8, w=30)

  pdf.set_font("Courier", style="B", size=14)
  pdf.cell(200, 10, txt=f"{ticker}", ln=True, align="C")
  pdf.ln(25)  # Space after header

  # Add stock chart image
  chart_url = f"https://charts2-node.finviz.com/chart.ashx?cs=l&t={ticker}&tf=d&s=linear&ct=candle_stick&tm=l&o[0][ot]=sma&o[0][op]=50&o[0][oc]=FF8F33C6&o[1][ot]=sma&o[1][op]=200&o[1][oc]=DCB3326D&o[2][ot]=sma&o[2][op]=20&o[2][oc]=DC32B363&o[3][ot]=patterns&o[3][op]=&o[3][oc]=000"
  chart_response = requests.get(chart_url, headers=headers)
  try:
    chart_image = Image.open(io.BytesIO(chart_response.content))
    chart_image_path = "chart.png"
    chart_image.save(chart_image_path)
    pdf.image(chart_image_path, x=10, y=pdf.get_y(), w=190)
    pdf.ln(80)  # Space after chart
  except Exception as e:
    st.error(f"Error loading chart image: {e}")

  # Add analysis sections
  for section, content in data.items():
    pdf.set_font("Courier", style="B", size=14)
    pdf.set_text_color(233, 82, 83)
    pdf.cell(200, 6, txt=section.replace("_", " ").capitalize(), ln=True, align="L")
    pdf.set_text_color(97, 98, 99)  # Reset to black for content
    pdf.set_font("Courier", size=12)
    for key, value in content.items():
      pdf.set_font("Courier", style="B", size=12)
      pdf.multi_cell(0, 6, txt=f"{key.replace('_', ' ').capitalize()}: ", align="L")
      pdf.set_font("Courier", size=12)
      pdf.multi_cell(0, 6, txt=f"{value}", align="L")
      pdf.ln(2)  # Space between key-value pairs
    pdf.ln(8)  # Space between sections

  return pdf.output(dest='S').encode('latin1')

if __name__ == "__main__":
  st.set_page_config(page_title="SEC Agent", page_icon=":material/history_edu:")
  st.title("SEC Agent :material/history_edu:")
  st.subheader("Agent that analyses companies 10-K or 10-Q filings")
  ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):", max_chars=5).upper()
  if st.button("Analyze Filing", type="primary"):
    with st.spinner("Analyzing..."):
      result, logo_url = sec_agent(ticker)
      col1, col2 = st.columns([1, 9], gap="small", vertical_alignment="center")
      with col1:
        st.image(logo_url, width=100)
      with col2:
        st.markdown(f"**[{ticker}](https://finviz.com/quote.ashx?t={ticker}&p=d)**")
      st.image(f"https://charts-node.finviz.com/chart.ashx?cs=l&t={ticker}&tf=d&s=linear&ct=candle_stick&tm=l&o[0][ot]=sma&o[0][op]=50&o[0][oc]=FF8F33C6&o[1][ot]=sma&o[1][op]=200&o[1][oc]=DCB3326D&o[2][ot]=sma&o[2][op]=20&o[2][oc]=DC32B363&o[3][ot]=patterns&o[3][op]=&o[3][oc]=000")
      # Iterar sobre las secciones del JSON
      for section, content in json.loads(result).items():
          # Formatear título de la sección
          section_title = section.replace("_", " ").capitalize()
          display_section(section_title, content)
          st.markdown("---")  # Línea divisoria entre secciones
      result_json = json.loads(result)

      # Crear y descargar el PDF
      pdf_data = create_pdf(result_json, ticker, logo_url)
      st.download_button(
          label="📥 Download Analysis Report as PDF",
          data=pdf_data,
          file_name=f"{ticker}_financial_analysis.pdf",
          mime="application/pdf"
      )