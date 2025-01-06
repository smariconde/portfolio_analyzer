import requests
from bs4 import BeautifulSoup
import re, json, io

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st
import yfinance as yf
from fpdf import FPDF


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
    st.json(data)

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
        return
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',  # Do Not Track Request Header
        'Connection': 'close'
    }
    response = requests.get(latest_filing, headers=headers)

    soup = BeautifulSoup(response.content, "html.parser")

    text_content = soup.get_text()
    # Create a prompt template for the analysis
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
            """Eres un experto inversor y analista financiero especializado en reportes 10-K y 10-Q. Tu tarea es analizar el reporte proporcionado y devolver un análisis completo en formato JSON. Cada sección debe ser breve y específica, basada en los datos del reporte. El formato es el siguiente (no uses markdown ni texto adicional):
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
                "key_metrics": "Indicadores financieros clave (ej. EPS, flujo de caja, etc.)."
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
                "investment_recommendation": "¿Es una buena inversión? Responde con 'sí' o 'no'.",
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

    return json_match.group(1)

# Función para formatear cada sección
def display_section(title, content):
    st.markdown(f"### {title}")
    for key, value in content.items():
        # Resaltar las claves y formatear valores
        st.markdown(f"**{key.replace('_', ' ').capitalize()}:** {value}")

# Crear el contenido del PDF
def create_pdf(data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for section, content in data.items():
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, txt=section.replace("_", " ").capitalize(), ln=True, align="L")
        pdf.set_font("Arial", size=12)
        for key, value in content.items():
            pdf.multi_cell(0, 10, txt=f"{key.replace('_', ' ').capitalize()}: {value}")
        pdf.ln(10)  # Espacio entre secciones
    return pdf.output(dest='S').encode('latin1') 

if __name__ == "__main__":
    st.title("SEC Agent")
    st.subheader("Agent that analyses companies 10-K or 10-Q filings")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):", max_chars=5, value="AAPL").upper()
    if st.button("Analyze Filing",type="primary"):
        result = sec_agent(ticker)
        # Iterar sobre las secciones del JSON
        for section, content in json.loads(result).items():
            # Formatear título de la sección
            section_title = section.replace("_", " ").capitalize()
            display_section(section_title, content)
            st.markdown("---")  # Línea divisoria entre secciones
        result_json = json.loads(result)

        # Crear y descargar el PDF
        pdf_data = create_pdf(result_json)
        st.download_button(
            label="📥 Download Analysis Report as PDF",
            data=pdf_data,
            file_name=f"financial_analysis_{ticker}.pdf",
            mime="application/pdf"
        )