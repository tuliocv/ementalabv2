# ===============================================================
# 🧠 EmentaLabv2 — Inteligência Curricular (v11.1)
# ===============================================================
import streamlit as st
import pandas as pd
from pathlib import Path
from openai import OpenAI

from utils.exportkit import _init_exports, export_zip_button
from utils.text_utils import normalize_text

# ---------------------------------------------------------------
# ⚙️ Configuração da Página
# ---------------------------------------------------------------
st.set_page_config(page_title="EmentaLabv2", layout="wide", page_icon="🧠")

# ---------------------------------------------------------------
# 🎨 SIDEBAR — Fluxo guiado por etapas
# ---------------------------------------------------------------
logo = Path("assets/logo.png")
if logo.exists():
    st.sidebar.image(str(logo), width=220)

st.sidebar.title("🧠 EmentaLabv2 — Inteligência Curricular")
st.sidebar.markdown("Ferramenta para análise automatizada de ementas e competências.")
st.sidebar.markdown("---")

# ===============================================================
# 1️⃣ ETAPA — Upload da Base Curricular
# ===============================================================
st.sidebar.subheader("📂 Etapa 1 — Carregar Base Curricular")
uploaded = st.sidebar.file_uploader("Envie o arquivo (.xlsx ou .csv)", type=["xlsx", "csv"])

if not uploaded:
    st.info("👈 Envie um arquivo Excel ou CSV na lateral para iniciar.")
    st.stop()

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

st.sidebar.success(f"✅ Base carregada ({len(df)} registros)")
st.sidebar.caption(f"📄 {uploaded.name} {sheet_info}")
st.sidebar.markdown("---")

# ===============================================================
# 2️⃣ ETAPA — Configuração da Chave OpenAI
# ===============================================================
st.sidebar.subheader("🔑 Etapa 2 — Chave OpenAI (opcional)")
api_key = st.sidebar.text_input(
    "Informe a chave (sk-...)", 
    type="password", 
    placeholder="sk-xxxxxxxxxxxxxxxx"
)
client = OpenAI(api_key=api_key) if api_key else None

if api_key:
    st.sidebar.success("✅ Chave validada com sucesso.")
else:
    st.sidebar.info("ℹ️ Sem chave: análises GPT ficarão desativadas.")
st.sidebar.markdown("---")

# ===============================================================
# 3️⃣ ETAPA — Aplicar Filtros
# ===============================================================
st.sidebar.subheader("🎯 Etapa 3 — Aplicar Filtros")

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

if active_filters:
    st.sidebar.success(f"🎯 {len(active_filters)} filtros aplicados")
else:
    st.sidebar.info("Nenhum filtro aplicado (todas as UCs incluídas).")

st.sidebar.caption(f"📊 Registros filtrados: {len(df_filtered)}")
st.sidebar.markdown("---")

# ===============================================================
# 4️⃣ ETAPA — Seleção do Tipo de Análise
# ===============================================================
st.sidebar.subheader("📈 Etapa 4 — Escolher Tipo de Análise")

menu = st.sidebar.selectbox(
    "Selecione uma análise:",
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

# Inicializa diretório de exportação
scope_key = normalize_text(menu).replace(" ", "_")
_init_exports(scope_key)

# ---------------------------------------------------------------
# CONTEÚDO PRINCIPAL — Cabeçalho e Filtros Ativos
# ---------------------------------------------------------------
st.markdown("## 🧩 EmentaLabv2 — Painel de Análise")
st.caption("Analise e explore relações entre ementas, objetivos, competências e coerência curricular.")
st.markdown("---")

with st.expander("🔍 Filtros aplicados", expanded=False):
    if active_filters:
        for k, v in active_filters.items():
            st.write(f"**{k}:** {', '.join(map(str, v))}")
    else:
        st.caption("Nenhum filtro aplicado.")

# ---------------------------------------------------------------
# EXECUÇÃO DAS ANÁLISES
# ---------------------------------------------------------------
if menu == "📊 Resumo Geral":
    from modules.summary_dashboard import run_summary
    st.header("📊 Resumo Geral")
    st.caption("Visão geral dos dados importados, número de UCs, cursos e distribuição geral.")
    run_summary(df_filtered, scope_key)

elif menu == "✅ Cobertura Curricular":
    from modules.coverage_report import run_coverage
    st.header("✅ Cobertura Curricular")
    st.caption("Mapeia o grau de cobertura das competências e conteúdos previstos nas UCs.")
    run_coverage(df_filtered, scope_key)

elif menu == "📈 Curva Bloom Progressiva":
    from modules.bloom_analysis import run_bloom
    st.header("📈 Curva Bloom Progressiva")
    st.caption("Analisa o nível cognitivo predominante (Taxonomia de Bloom) dos objetivos de aprendizagem.")
    run_bloom(df_filtered, scope_key, client)

elif menu == "🎯 Alinhamento de Objetivos e Competências":
    from modules.alignment_topk import run_alignment
    st.header("🎯 Alinhamento de Objetivos e Competências")
    st.caption("Avalia a coerência entre os objetivos de aprendizagem e as competências do egresso.")
    run_alignment(df_filtered, scope_key)

elif menu == "🧩 Similaridade e Redundância":
    from modules.redundancy_matrix import run_redundancy, run_pair_analysis
    st.header("🧩 Similaridade e Redundância")
    st.caption("Detecta sobreposições de conteúdo entre ementas e permite comparar UCs frase a frase.")
    tab1, tab2 = st.tabs(["🔁 Redundância entre UCs", "🔬 Comparação Frase a Frase"])
    with tab1:
        run_redundancy(df_filtered, scope_key)
    with tab2:
        run_pair_analysis(df_filtered, scope_key)

elif menu == "🌐 Convergência Temática":
    from modules.clusterization import run_cluster
    st.header("🌐 Convergência Temática")
    st.caption("Agrupa UCs com base na similaridade semântica de seus conteúdos, permitindo identificar convergências interdisciplinares.")
    run_cluster(df_filtered, scope_key, client)

elif menu == "🔗 Dependência Curricular":
    from modules.dependency_graph import run_graph
    st.header("🔗 Dependência Curricular")
    st.caption("Identifica relações de precedência e interdependência entre UCs, com base em similaridade e inferência semântica.")
    run_graph(df_filtered, scope_key, client)

elif menu == "💬 Clareza e Sentimento das Ementas":
    from modules.sentiment_analysis import run_sentiment
    st.header("💬 Clareza e Sentimento das Ementas")
    st.caption("Analisa o tom e a clareza textual das ementas, detectando vieses ou falta de objetividade.")
    run_sentiment(df_filtered, scope_key, client)

elif menu == "📆 Análise Longitudinal":
    from modules.longitudinal_analysis import run_longitudinal
    st.header("📆 Análise Longitudinal")
    st.caption("Acompanha revisões e evoluções curriculares ao longo dos semestres ou versões das ementas.")
    run_longitudinal(df_filtered, scope_key, client)

elif menu == "🤖 Relatório Consultivo":
    from modules.consultive_report import run_consultive
    st.header("🤖 Relatório Consultivo")
    st.caption("Gera um relatório automatizado com diagnósticos e recomendações sobre a coerência curricular geral.")
    run_consultive(df_filtered, scope_key, client)

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
