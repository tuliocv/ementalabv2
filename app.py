# ===============================================================
# 🧠 EmentaLabv2 — Inteligência Curricular (v10.3)
# ===============================================================

import streamlit as st
import pandas as pd
from pathlib import Path
from openai import OpenAI

from utils.exportkit import _init_exports, export_zip_button
from utils.text_utils import normalize_text

# ---------------------------------------------------------------
# ⚙️ Configuração de Página
# ---------------------------------------------------------------
st.set_page_config(page_title="EmentaLabv2", layout="wide", page_icon="🧠")

# ---------------------------------------------------------------
# 🎨 Sidebar — Identidade e Configurações
# ---------------------------------------------------------------
logo = Path("assets/logo.png")
if logo.exists():
    st.sidebar.image(str(logo), width=220)

st.sidebar.title("🧠 EmentaLabv2 — Inteligência Curricular")
st.sidebar.markdown("---")

# 🔑 Configuração global de API Key
st.sidebar.subheader("🔑 Configurações Globais")
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Informe sua chave da OpenAI (opcional). Ela será usada nos módulos que utilizam GPT."
)
client = OpenAI(api_key=api_key) if api_key else None
if api_key:
    st.sidebar.success("✅ Chave carregada com sucesso.")
else:
    st.sidebar.info("ℹ️ Insira a API Key apenas se desejar usar recursos de IA.")
st.sidebar.markdown("---")

# ---------------------------------------------------------------
# 📂 Upload de Arquivo
# ---------------------------------------------------------------
st.sidebar.header("📂 Base Curricular")
uploaded = st.sidebar.file_uploader("Carregar arquivo (.xlsx ou .csv)", type=["xlsx", "csv"])
if not uploaded:
    st.info("👈 Envie um arquivo Excel ou CSV para iniciar a análise.")
    st.stop()

try:
    df = pd.read_excel(uploaded, engine="openpyxl") if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

# ---------------------------------------------------------------
# 🎯 Filtros
# ---------------------------------------------------------------
st.sidebar.subheader("🎯 Filtros")
for col in ["Nome do curso", "Modalidade do curso", "Tipo Graduação", "Cluster", "Tipo do componente"]:
    if col in df.columns:
        sel = st.sidebar.multiselect(col, sorted(df[col].dropna().unique()))
        if sel:
            df = df[df[col].isin(sel)]

# ---------------------------------------------------------------
# 🧭 Menu de Análises
# ---------------------------------------------------------------
menu = st.sidebar.selectbox(
    "Selecione a análise desejada:",
    [
        "📊 Resumo Geral",
        "✅ Cobertura Curricular",
        "📈 Curva Bloom Progressiva",
        "🎯 Alinhamento de Objetivos e Competências",
        "🧩 Similaridade e Redundância",
        "🌐 Convergência Temática",
        "🔗 Dependência Curricular",
        "💬 Clareza e Sentimento das Ementas",
        "📆 Análise Longitudinal",
        "🤖 Relatório Consultivo"
    ]
)

# Normaliza e cria escopo de exportação
scope_key = normalize_text(menu).replace(" ", "_")
_init_exports(scope_key)

# ---------------------------------------------------------------
# 🚀 Roteamento por Tipo de Análise
# ---------------------------------------------------------------
if menu == "📊 Resumo Geral":
    from modules.summary_dashboard import run_summary
    run_summary(df, scope_key)

elif menu == "✅ Cobertura Curricular":
    from modules.coverage_report import run_coverage
    run_coverage(df, scope_key)

elif menu == "📈 Curva Bloom Progressiva":
    from modules.bloom_analysis import run_bloom
    run_bloom(df, scope_key, client)

elif menu == "🎯 Alinhamento de Objetivos e Competências":
    from modules.alignment_topk import run_alignment
    run_alignment(df, scope_key)

elif menu == "🧩 Similaridade e Redundância":
    from modules.redundancy_matrix import run_redundancy
    run_redundancy(df, scope_key)

elif menu == "🌐 Convergência Temática":
    from modules.clusterization import run_cluster
    run_cluster(df, scope_key, client)

elif menu == "🔗 Dependência Curricular":
    from modules.dependency_graph_interactive import run_graph_interactive
    run_graph_interactive(df, scope_key, client)

elif menu == "💬 Clareza e Sentimento das Ementas":
    from modules.sentiment_analysis import run_sentiment
    run_sentiment(df, scope_key, client)

elif menu == "📆 Análise Longitudinal":
    from modules.longitudinal_analysis import run_longitudinal
    run_longitudinal(df, scope_key, client)

elif menu == "🤖 Relatório Consultivo":
    from modules.consultive_report import run_consultive
    run_consultive(df, scope_key, client)

# ---------------------------------------------------------------
# 📦 Exportação Global
# ---------------------------------------------------------------
st.markdown("---")
export_zip_button(scope_key)

# ---------------------------------------------------------------
# 🧭 Rodapé
# ---------------------------------------------------------------
st.markdown("---")
st.caption("""
📘 **EmentaLabv2** — Ferramenta de análise curricular inteligente.  
Apoia NDEs e coordenações na revisão de coerência, progressão cognitiva e integração pedagógica das Unidades Curriculares.  
Desenvolvido com 💙 e IA aplicada à educação. :)
""")
