# ===============================================================
# 🧠 EmentaLabv2 — Inteligência Curricular (v10.5)
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
# 🎨 Sidebar — Identidade e Configuração
# ---------------------------------------------------------------
logo = Path("assets/logo.png")
if logo.exists():
    st.sidebar.image(str(logo), width=220)

st.sidebar.title("🧠 EmentaLabv2 — Inteligência Curricular")
st.sidebar.markdown("---")

# 🔑 API key global (usada por módulos com GPT)
st.sidebar.subheader("🔑 Configurações")
api_key = st.sidebar.text_input(
    "OpenAI API Key (opcional)",
    type="password",
    placeholder="sk-...",
    help="Se informada, será usada nos módulos que utilizam GPT (Bloom, Clusters, Dependência, Relatório etc.)"
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
    st.info("👈 Envie um arquivo Excel ou CSV para iniciar.")
    st.stop()

# leitura segura
try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
        sheet_info = "(CSV)"
    else:
        df = pd.read_excel(uploaded, engine="openpyxl")
        sheet_info = "(XLSX)"
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

# limpeza de colunas
df.columns = (
    df.columns
    .str.replace(r"[\n\r\xa0]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

st.caption(f"📄 Arquivo: **{uploaded.name}** {sheet_info} | Registros: **{len(df)}**")

# ---------------------------------------------------------------
# 🎯 Filtros Essenciais
# ---------------------------------------------------------------
st.sidebar.subheader("🎯 Filtros")
filter_cols = ["Nome do curso", "Modalidade do curso", "Tipo Graduação", "Cluster", "Tipo do componente"]
df_filtered = df.copy()
active_filters = {}

for col in filter_cols:
    if col in df.columns:
        values = sorted(df[col].dropna().astype(str).unique())
        sel = st.sidebar.multiselect(col, values, default=[])
        if sel:
            df_filtered = df_filtered[df_filtered[col].astype(str).isin(sel)]
            active_filters[col] = sel

# ---------------------------------------------------------------
# 🧭 Menu de Análises (nomes institucionais)
# ---------------------------------------------------------------
menu = st.sidebar.selectbox(
    "Tipo de análise",
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
        "🤖 Relatório Consultivo",
    ],
    index=0
)

# define escopo e inicializa exportações
scope_key = normalize_text(menu).replace(" ", "_")
_init_exports(scope_key)

# ---------------------------------------------------------------
# 🔍 Contexto do Filtro
# ---------------------------------------------------------------
with st.expander("🔍 Contexto dos filtros aplicados", expanded=False):
    if active_filters:
        for k, v in active_filters.items():
            st.write(f"**{k}:** {', '.join(map(str, v))}")
    else:
        st.caption("Nenhum filtro aplicado.")

st.markdown("---")

# ---------------------------------------------------------------
# 🚀 Roteamento por Tipo de Análise
# ---------------------------------------------------------------
if menu == "📊 Resumo Geral":
    try:
        from modules.summary_dashboard import run_summary
        st.header("📊 Resumo Geral")
        st.caption("Visão geral dos dados importados, número de UCs, cursos e distribuição geral.")
        run_summary(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro ao carregar Resumo Geral: {e}")

elif menu == "✅ Cobertura Curricular":
    try:
        from modules.coverage_report import run_coverage
        st.header("✅ Cobertura Curricular")
        st.caption("Mapeia o grau de cobertura das competências e conteúdos previstos nas UCs.")
        run_coverage(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro na Cobertura Curricular: {e}")

elif menu == "📈 Curva Bloom Progressiva":
    try:
        from modules.bloom_analysis import run_bloom
        st.header("📈 Curva Bloom Progressiva")
        st.caption("Analisa o nível cognitivo predominante (Taxonomia de Bloom) dos objetivos de aprendizagem.")
        run_bloom(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro na Curva Bloom Progressiva: {e}")

elif menu == "🎯 Alinhamento de Objetivos e Competências":
    try:
        from modules.alignment_topk import run_alignment
        st.header("🎯 Alinhamento de Objetivos e Competências")
        st.caption("Avalia a coerência entre os objetivos de aprendizagem e as competências do egresso.")
        run_alignment(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro no Alinhamento: {e}")

elif menu == "🧩 Similaridade e Redundância":
    try:
        from modules.redundancy_matrix import run_redundancy, run_pair_analysis
        st.header("🧩 Similaridade e Redundância")
        st.caption("Detecta sobreposições de conteúdo entre ementas e permite comparar UCs frase a frase.")
        tab1, tab2 = st.tabs(["🔁 Redundância entre UCs", "🔬 Comparação Frase a Frase"])
        with tab1:
            run_redundancy(df_filtered, scope_key)
        with tab2:
            run_pair_analysis(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro em Similaridade e Redundância: {e}")

elif menu == "🌐 Convergência Temática":
    try:
        from modules.clusterization import run_cluster
        st.header("🌐 Convergência Temática")
        st.caption("Agrupa UCs com base na similaridade semântica de seus conteúdos, permitindo identificar convergências interdisciplinares.")
        run_cluster(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro na Convergência Temática: {e}")

elif menu == "🔗 Dependência Curricular":
    try:
        from modules.dependency_graph import run_graph
        st.header("🔗 Dependência Curricular")
        st.caption("Identifica relações de precedência e interdependência entre UCs, com base em similaridade e inferência semântica.")
        run_graph(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro na Dependência Curricular: {e}")

elif menu == "💬 Clareza e Sentimento das Ementas":
    try:
        from modules.sentiment_analysis import run_sentiment
        st.header("💬 Clareza e Sentimento das Ementas")
        st.caption("Analisa o tom e a clareza textual das ementas, detectando vieses ou falta de objetividade.")
        run_sentiment(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro em Clareza e Sentimento: {e}")

elif menu == "📆 Análise Longitudinal":
    try:
        from modules.longitudinal_analysis import run_longitudinal
        st.header("📆 Análise Longitudinal")
        st.caption("Acompanha revisões e evoluções curriculares ao longo dos semestres ou versões das ementas.")
        run_longitudinal(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro na Análise Longitudinal: {e}")

elif menu == "🤖 Relatório Consultivo":
    try:
        from modules.consultive_report import run_consultive
        st.header("🤖 Relatório Consultivo")
        st.caption("Gera um relatório automatizado com diagnósticos e recomendações sobre a coerência curricular geral.")
        run_consultive(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro no Relatório Consultivo: {e}")

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
Desenvolvido para apoiar **NDEs e coordenações** na revisão de coerência, progressão cognitiva e integração pedagógica.
""")
