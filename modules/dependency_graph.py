# ===============================================================
# 🔗 EmentaLabv2 — Sequenciamento / Grafo (GPT)
# ===============================================================
import streamlit as st
import pandas as pd
from openai import OpenAI
from utils.text_utils import find_col, truncate
from utils.exportkit import export_table, export_zip_button
import networkx as nx
import matplotlib.pyplot as plt

def run_graph(df, scope_key):
    col_obj = find_col(df, "Objetos de conhecimento")
    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' não encontrada.")
        st.stop()

    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    if not api_key:
        st.stop()

    client = OpenAI(api_key=api_key)
    subset = df[["Nome da UC", col_obj]].dropna().head(10)
    prompt = "Identifique relações de pré-requisito entre UCs:\n"
    for _, r in subset.iterrows():
        prompt += f"- {r['Nome da UC']}: {truncate(r[col_obj])}\n"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    content = resp.choices[0].message.content
    st.code(content[:800], language="json")

    st.info("Visualização e parsing avançado estão desativados nesta versão modular.")
    export_zip_button(scope_key)
