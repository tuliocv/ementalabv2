import streamlit as st
import pandas as pd
from pathlib import Path

from utils.exportkit import _init_exports, export_zip_button
from utils.text_utils import normalize_text

from modules.dependency_graph_interactive import run_graph_interactive



# Configuração de página
st.set_page_config(page_title="EmentaLabv2", layout="wide", page_icon="🧠")

# Logo e título
logo = Path("assets/logo.png")
if logo.exists():
    st.sidebar.image(str(logo), width=220)
st.sidebar.title("🧠 EmentaLabv2 — Inteligência Curricular")

# Upload
uploaded = st.sidebar.file_uploader("📂 Carregar base curricular (.xlsx ou .csv)", type=["xlsx", "csv"])
if not uploaded:
    st.info("👈 Envie um arquivo Excel ou CSV para iniciar.")
    st.stop()

try:
    df = pd.read_excel(uploaded, engine="openpyxl") if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Erro ao carregar: {e}")
    st.stop()

# Filtros
st.sidebar.subheader("🎯 Filtros")
for col in ["Nome do curso", "Modalidade do curso", "Tipo Graduação", "Cluster", "Tipo do componente"]:
    if col in df.columns:
        sel = st.sidebar.multiselect(col, sorted(df[col].dropna().unique()))
        if sel:
            df = df[df[col].isin(sel)]

# Menu
menu = st.sidebar.selectbox("Selecione a análise:", [
    "📊 Resumo",
    "🔎 Relatório de Cobertura",
    "🧩 Similaridade (Objetos × Comp & DCN)",
    "🎯 Alinhamento (Objetivos × Competências)",
    "🧬 Redundância entre UCs",
    "🔬 Análise Pontual (Frase a Frase)",
    "🧠 Mapa de Bloom (Heurística + GPT)",
    "📈 Clusterização (Ementa)",
    "🔗 Sequenciamento / Grafo (GPT)",
    "🤖 Relatório Consultivo (GPT)"
])

scope_key = normalize_text(menu).replace(" ", "_")
_init_exports(scope_key)

# Chamada dinâmica
if menu == "📊 Resumo":
    from modules.summary_dashboard import run_summary
    run_summary(df, scope_key)

elif menu == "🔎 Relatório de Cobertura":
    from modules.coverage_report import run_coverage
    run_coverage(df, scope_key)

elif menu == "🧩 Similaridade (Objetos × Comp & DCN)":
    from modules.similarity_matrix import run_similarity
    run_similarity(df, scope_key)

elif menu == "🎯 Alinhamento (Objetivos × Competências)":
    from modules.alignment_topk import run_alignment
    run_alignment(df, scope_key)

elif menu == "🧬 Redundância entre UCs":
    from modules.redundancy_matrix import run_redundancy
    run_redundancy(df, scope_key)

elif menu == "🔬 Análise Pontual (Frase a Frase)":
    from modules.redundancy_matrix import run_pair_analysis
    run_pair_analysis(df, scope_key)

elif menu == "🧠 Mapa de Bloom (Heurística + GPT)":
    from modules.bloom_analysis import run_bloom
    run_bloom(df, scope_key)

elif menu == "📈 Clusterização (Ementa)":
    from modules.clusterization import run_cluster
    run_cluster(df, scope_key)

elif menu == "🔗 Sequenciamento / Grafo (GPT)":
    from modules.dependency_graph import run_graph
    run_graph(df, scope_key)

elif menu == "🤖 Relatório Consultivo (GPT)":
    from modules.consultive_report import run_consultive
    run_consultive(df, scope_key)

# Dentro da escolha de análises:
elif analise == "Grafo Interativo (PyVis)":
    run_graph_interactive(df, scope_key)
