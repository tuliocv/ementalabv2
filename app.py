# ===============================================================
# EmentaLabv2 — Inteligência Curricular Modular
# ===============================================================
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------
st.set_page_config(
    page_title="EmentaLabv2 — Inteligência Curricular",
    page_icon="🧠",
    layout="wide"
)

st.sidebar.image("assets/logo.png", width=220)
st.sidebar.title("🧭 EmentaLabv2")

# ---------------------------------------------------------------
# Upload da base
# ---------------------------------------------------------------
uploaded = st.sidebar.file_uploader("📂 Carregar base curricular (.xlsx ou .csv)", type=["xlsx", "csv"])
if not uploaded:
    st.stop()

df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)

menu = st.sidebar.selectbox(
    "Selecione a análise:",
    [
        "Resumo",
        "Cobertura por Competência",
        "Curva de Bloom Progressiva",
        "Convergência Temática",
        "Dependência Curricular",
        "Sentimento e Clareza",
        "Análise Longitudinal"
    ]
)

# ---------------------------------------------------------------
# Execução dos módulos
# ---------------------------------------------------------------
if menu == "Resumo":
    st.header("📋 Resumo da Base Curricular")
    st.dataframe(df.head(), use_container_width=True)

elif menu == "Cobertura por Competência":
    from modules.coverage_analysis import coverage_analysis
    coverage_analysis(df)

elif menu == "Curva de Bloom Progressiva":
    from modules.bloom_progressive import bloom_progressive
    bloom_progressive(df)

elif menu == "Convergência Temática":
    from modules.thematic_convergence import thematic_convergence
    thematic_convergence(df)

elif menu == "Dependência Curricular":
    from modules.dependency_graph import dependency_graph
    dependency_graph(df)

elif menu == "Sentimento e Clareza":
    from modules.sentiment_clarity import sentiment_clarity
    sentiment_clarity(df)

elif menu == "Análise Longitudinal":
    from modules.longitudinal_analysis import longitudinal_analysis
    df_antigo = st.file_uploader("📂 Carregar versão anterior (.xlsx)", type=["xlsx"])
    if df_antigo:
        df_old = pd.read_excel(df_antigo)
        longitudinal_analysis(df_old, df)
