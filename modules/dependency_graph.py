# ===============================================================
# 🔗 EmentaLabv2 — Sequenciamento / Grafo (GPT + Visual)
# ===============================================================
import streamlit as st
import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
from openai import OpenAI
from utils.text_utils import find_col, truncate
from utils.exportkit import export_zip_button, export_table

def parse_dependencies(text: str):
    """
    Extrai pares UC_A -> UC_B a partir do texto GPT.
    Detecta padrões como:
    - "X é pré-requisito de Y"
    - "Y depende de X"
    """
    deps = []
    lines = text.split("\n")
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        match1 = re.findall(r"(.+?)\s+(?:é|são)\s+pré-?requisito[s]?\s+(?:de|para)\s+(.+)", ln, flags=re.I)
        match2 = re.findall(r"(.+?)\s+depende\s+(?:de|do)\s+(.+)", ln, flags=re.I)
        for a, b in match1 + match2:
            a, b = a.strip(" .-–:;"), b.strip(" .-–:;")
            if len(a.split()) <= 10 and len(b.split()) <= 10:
                deps.append((a, b))
    return deps

def draw_graph(pairs):
    """Cria grafo simples com NetworkX"""
    G = nx.DiGraph()
    G.add_edges_from(pairs)

    pos = nx.spring_layout(G, k=0.7, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color="#a5d8ff", node_size=2000, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="#3b5bdb", width=2)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_color="#1c1c1c")
    ax.set_axis_off()
    plt.tight_layout()
    return fig

def run_graph(df, scope_key):
    st.header("🔗 Dependência Curricular — Relações de Pré-requisito entre UCs")
    st.markdown("""
**Objetivo:** identificar relações de precedência entre as Unidades Curriculares (UCs),
baseando-se nos conteúdos e competências descritos nas ementas.
    """)

    col_obj = find_col(df, "Objetos de conhecimento")
    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' não encontrada.")
        st.stop()

    # API Key
    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    if not api_key:
        st.stop()

    client = OpenAI(api_key=api_key)
    subset = df[["Nome da UC", col_obj]].dropna().head(10)

    prompt = (
        "Analise as UCs abaixo e identifique as relações de pré-requisito entre elas. "
        "Liste em formato textual claro, por exemplo:\n"
        "- X é pré-requisito de Y\n\n"
    )
    for _, r in subset.iterrows():
        prompt += f"- {r['Nome da UC']}: {truncate(r[col_obj])}\n"

    with st.spinner("🧠 Consultando o modelo GPT..."):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":]()
