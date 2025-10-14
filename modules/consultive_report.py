# ===============================================================
# 🤖 EmentaLabv2 — Relatório Consultivo GPT
# ===============================================================
import streamlit as st
from openai import OpenAI
from utils.exportkit import export_zip_button, get_docx_bytes

def run_consultive(df, scope_key):
    st.header("🤖 Relatório Consultivo (GPT)")
    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    if not api_key:
        st.stop()

    client = OpenAI(api_key=api_key)
    st.info("Gerando relatório resumido sobre coerência curricular...")

    prompt = (
        "Você é um consultor educacional. Gere um diagnóstico breve sobre coerência curricular, "
        "nível cognitivo predominante e alinhamento com perfil do egresso com base na planilha fornecida."
    )
    resp = client.chat.completions.create(
        model="gpt-
