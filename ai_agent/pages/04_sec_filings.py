import io
import json
import os
import re
import sys
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ai_agent.env_loader import load_project_env
from ai_agent.llm_utils import invoke_gemini

load_project_env()
import yfinance as yf
from bs4 import BeautifulSoup
from fpdf import FPDF
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
    "Connection": "close",
}


def _extract_json_payload(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty model response.")

    # 1) Direct parse.
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 2) Parse from fenced JSON block if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # 3) Parse the outermost balanced JSON object.
    start = raw.find("{")
    if start != -1:
        depth = 0
        for idx, char in enumerate(raw[start:], start=start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    block = raw[start : idx + 1]
                    try:
                        data = json.loads(block)
                        if isinstance(data, dict):
                            return data
                    except json.JSONDecodeError:
                        break

    raise ValueError("Model returned no valid JSON block.")


def sec_agent(ticker):
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY. Set it in your environment or .env file.")

    stock = yf.Ticker(ticker)
    filings = stock.sec_filings
    website = stock.info.get("website", "")
    domain = website.replace("http://", "").replace("https://", "").split("/")[0]
    logo_url = f"https://logo.clearbit.com/{domain}" if domain else None

    latest_filing = None
    for item in filings:
        if isinstance(item, dict) and item.get("type") in {"10-K", "10-Q"}:
            latest_filing = item.get("exhibits", {}).get(item["type"])
            if latest_filing:
                break

    if latest_filing is None:
        raise RuntimeError("No 10-K or 10-Q filing found for this ticker.")

    response = requests.get(latest_filing, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    text_content = re.sub(r"\s+", " ", soup.get_text()).strip()
    max_chars = int(os.environ.get("SEC_MAX_INPUT_CHARS", "120000"))
    text_content = text_content[:max_chars]

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Eres un experto inversor y analista financiero especializado en reportes 10-K y 10-Q.
Devuelve un análisis completo en formato JSON con secciones: filing, overview,
financial_performance, risks, strategies_future, projections, management,
corporate_actions, conclusion. No uses markdown ni texto fuera del JSON.""",
            ),
            (
                "human",
                f"Analiza este reporte y responde solo JSON: {text_content}",
            ),
        ]
    )
    result, model_used = invoke_gemini(
        template.invoke({"text_content": text_content}),
        temperature=0,
        max_tokens=None,
        max_retries=6,
        stop=None,
    )
    raw_output = result.content

    try:
        result_json = _extract_json_payload(raw_output)
    except ValueError:
        repair_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Return only valid JSON object. No markdown. No explanations.",
                ),
                (
                    "human",
                    "Convert this SEC analysis output to valid JSON with these keys only: "
                    "filing, overview, financial_performance, risks, strategies_future, "
                    "projections, management, corporate_actions, conclusion.\n\n"
                    f"Text:\n{raw_output}",
                ),
            ]
        )
        repaired, _ = invoke_gemini(
            repair_prompt.invoke({}),
            temperature=0,
            max_tokens=None,
            max_retries=4,
            stop=None,
        )
        result_json = _extract_json_payload(repaired.content)

    return result_json, logo_url, model_used


def display_section(title, content):
    st.markdown(f"### {title}")
    if isinstance(content, dict):
        for key, value in content.items():
            st.markdown(f"**{key.replace('_', ' ').capitalize()}:** {value}")
    else:
        st.write(content)


def create_pdf(data, ticker, logo_url):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Courier", size=12)

    if logo_url:
        try:
            logo_response = requests.get(logo_url, timeout=20)
            logo_response.raise_for_status()
            logo_image = Image.open(io.BytesIO(logo_response.content))
            logo_image_path = "logo.png"
            logo_image.save(logo_image_path)
            pdf.image(logo_image_path, x=10, y=8, w=30)
        except Exception:
            pass

    pdf.set_font("Courier", style="B", size=14)
    pdf.cell(200, 10, txt=f"{ticker}", ln=True, align="C")
    pdf.ln(25)

    for section, content in data.items():
        pdf.set_font("Courier", style="B", size=14)
        pdf.set_text_color(233, 82, 83)
        pdf.cell(200, 6, txt=section.replace("_", " ").capitalize(), ln=True, align="L")
        pdf.set_text_color(97, 98, 99)
        pdf.set_font("Courier", size=12)
        if isinstance(content, dict):
            for key, value in content.items():
                pdf.set_font("Courier", style="B", size=12)
                pdf.multi_cell(0, 6, txt=f"{key.replace('_', ' ').capitalize()}: ", align="L")
                pdf.set_font("Courier", size=12)
                pdf.multi_cell(0, 6, txt=f"{value}", align="L")
                pdf.ln(2)
        else:
            pdf.multi_cell(0, 6, txt=str(content), align="L")
        pdf.ln(8)

    return pdf.output(dest="S").encode("latin1")


@st.cache_data(ttl=21600)
def load_sec_cached(ticker: str):
    return sec_agent(ticker)


if __name__ in {"__main__", "__page__"}:
    st.title("SEC Agent :material/history_edu:")
    st.caption("Analyze latest 10-K / 10-Q filing and generate a downloadable report.")
    st.info(
        "The model summarizes the latest filing available via Yahoo Finance SEC links.",
        icon=":material/info:",
    )

    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning("`GOOGLE_API_KEY` is missing. SEC analysis requests will fail.")
    if st.button("Refresh SEC cache", icon=":material/refresh:"):
        st.cache_data.clear()
        st.rerun()

    with st.form("sec_form"):
        ticker = st.text_input("Ticker Symbol", max_chars=8, placeholder="AAPL").upper().strip()
        submit = st.form_submit_button("Analyze Filing", type="primary")

    if submit:
        if not ticker:
            st.error("Enter a valid ticker symbol.")
            st.stop()

        with st.spinner("Analyzing filing..."):
            try:
                result_json, logo_url, model_used = load_sec_cached(ticker)
            except Exception as exc:
                st.error(f"SEC analysis failed: {exc}")
                st.stop()
        st.caption(f"Model used: `{model_used}`")

        col1, col2 = st.columns([1, 9], gap="small", vertical_alignment="center")
        with col1:
            if logo_url:
                st.image(logo_url, width=100)
        with col2:
            st.markdown(f"**[{ticker}](https://finviz.com/quote.ashx?t={ticker}&p=d)**")

        st.image(
            "https://charts-node.finviz.com/chart.ashx"
            f"?cs=l&t={ticker}&tf=d&s=linear&ct=candle_stick&tm=l"
            "&o[0][ot]=sma&o[0][op]=50&o[0][oc]=FF8F33C6"
            "&o[1][ot]=sma&o[1][op]=200&o[1][oc]=DCB3326D"
            "&o[2][ot]=sma&o[2][op]=20&o[2][oc]=DC32B363"
            "&o[3][ot]=patterns&o[3][op]=&o[3][oc]=000"
        )

        tab_sections, tab_raw = st.tabs(["Report", "Raw JSON"])
        with tab_sections:
            for section, content in result_json.items():
                section_title = section.replace("_", " ").capitalize()
                display_section(section_title, content)
                st.markdown("---")
        with tab_raw:
            st.json(result_json)

        pdf_data = create_pdf(result_json, ticker, logo_url)
        st.download_button(
            label="Download Analysis Report (PDF)",
            data=pdf_data,
            file_name=f"{ticker}_financial_analysis.pdf",
            mime="application/pdf",
        )
