# ===============================================================
# EmentaLabv2 — Inteligência Curricular Modular (v8.2)
# ===============================================================

import streamlit as st
import pandas as pd
import openpyxl
from pathlib import Path

# ---------------------------------------------------------------
# 1. CONFIGURAÇÕES INICIAIS
# ---------------------------------------------------------------
st.set_page_config(
    page_title="EmentaLabv2 — Inteligência Curricular",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------------
# 2. LOGO / CABEÇALHO
# ---------------------------------------------------------------
logo_path = Path("assets/logo.png")
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=220)
else:
    st.sidebar.markdown("## 🧠 EmentaLabv2")

st.sidebar.title("🧭 Navegação")

# ---------------------------------------------------------------
# 3. UPLOAD DO ARQUIVO
# ---------------------------------------------------------------
uploaded = st.sidebar.file_uploader("📂 Carregar base curricular (.xlsx ou .csv)", type=["xlsx", "csv"])

if not uploaded:
    st.info("👈 Envie um arquivo Excel ou CSV para iniciar a análise.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, engine="openpyxl")
except Exception as e:
    st.error(f"❌ Erro ao carregar o arquivo: {e}")
    st.stop()

# ---------------------------------------------------------------
# 4. FILTROS DE CONTEXTO
# ---------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filtros de Contexto")

if "Nome do curso" in df.columns:
    cursos = sorted(df["Nome do curso"].dropna().unique().tolist())
    curso_sel = st.sidebar.multiselect("📘 Nome do Curso", cursos)
    if curso_sel:
        df = df[df["Nome do curso"].isin(curso_sel)]

if "Modalidade do curso" in df.columns:
    modalidades = sorted(df["Modalidade do curso"].dropna().unique().tolist())
    mod_sel = st.sidebar.multiselect("🏫 Modalidade do Curso", modalidades)
    if mod_sel:
        df = df[df["Modalidade do curso"].isin(mod_sel)]

if "Tipo Graduação" in df.columns:
    tipos = sorted(df["Tipo Graduação"].dropna().unique().tolist())
    tipo_sel = st.sidebar.multiselect("🎓 Tipo de Graduação", tipos)
    if tipo_sel:
        df = df[df["Tipo Graduação"].isin(tipo_sel)]

if "Cluster" in df.columns:
    clusters = sorted(df["Cluster"].dropna().unique().tolist())
    cluster_sel = st.sidebar.multiselect("🌐 Cluster", clusters)
    if cluster_sel:
        df = df[df["Cluster"].isin(cluster_sel)]

if "Tipo do componente" in df.columns:
    tipos_comp = sorted(df["Tipo do componente"].dropna().unique().tolist())
    tipo_comp_sel = st.sidebar.multiselect("🧩 Tipo do Componente", tipos_comp)
    if tipo_comp_sel:
        df = df[df["Tipo do componente"].isin(tipo_comp_sel)]

st.sidebar.markdown("---")
st.sidebar.info(f"📊 {len(df)} registros após aplicação dos filtros.")

# ---------------------------------------------------------------
# 5. MENU PRINCIPAL DE ANÁLISES
# ---------------------------------------------------------------
menu = st.sidebar.selectbox(
    "Selecione a análise:",
    [
        "📋 Resumo da Base",
        "🧩 Cobertura por Competência",
        "📈 Curva de Bloom Progressiva",
        "🌐 Convergência Temática",
        "🔗 Dependência Curricular",
        "✍️ Sentimento e Clareza Linguística",
        "🧮 Análise Longitudinal (Versões)"
    ]
)

# ---------------------------------------------------------------
# 6. EXECUÇÃO DOS MÓDULOS
# ---------------------------------------------------------------
if menu == "📋 Resumo da Base":
    st.header("📋 Resumo da Base Curricular")
    st.dataframe(df.head(), use_container_width=True)
    st.success(f"Base carregada com {len(df)} registros e {len(df.columns)} colunas.")

elif menu == "🧩 Cobertura por Competência":
    from modules.coverage_analysis import coverage_analysis
    coverage_analysis(df)

elif menu == "📈 Curva de Bloom Progressiva":
    from modules.bloom_progressive import bloom_progressive
    bloom_progressive(df)

elif menu == "🌐 Convergência Temática":
    from modules.thematic_convergence import thematic_convergence
    thematic_convergence(df)

elif menu == "🔗 Dependência Curricular":
    from modules.dependency_graph import dependency_graph
    dependency_graph(df)

elif menu == "✍️ Sentimento e Clareza Linguística":
    from modules.sentiment_clarity import sentiment_clarity
    sentiment_clarity(df)

elif menu == "🧮 Análise Longitudinal (Versões)":
    from modules.longitudinal_analysis import longitudinal_analysis
    df_antigo = st.file_uploader("📂 Carregar versão anterior (.xlsx)", type=["xlsx"])
    if df_antigo:
        try:
            df_old = pd.read_excel(df_antigo, engine="openpyxl")
            longitudinal_analysis(df_old, df)
        except Exception as e:
            st.error(f"Erro ao carregar versão anterior: {e}")
