# ===============================================================
# 🧠 EmentaLabv2 — Inteligência Curricular (v11.0)
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
# 🎨 Sidebar — Identidade e Configuração
# ---------------------------------------------------------------
logo = Path("assets/logo.png")
if logo.exists():
    st.sidebar.image(str(logo), width=220)

st.sidebar.title("🧠 EmentaLabv2 — Inteligência Curricular")
st.sidebar.markdown("---")

# ===============================================================
# 🧩 ETAPA 1 — Upload do Arquivo
# ===============================================================
st.sidebar.header("📂 Etapa 1 — Carregar Base Curricular")
uploaded = st.sidebar.file_uploader("Selecione o arquivo (.xlsx ou .csv)", type=["xlsx", "csv"])

if not uploaded:
    st.info("👈 Envie um arquivo Excel ou CSV para iniciar a análise.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
        tipo_arquivo = "CSV"
    else:
        df = pd.read_excel(uploaded, engine="openpyxl")
        tipo_arquivo = "Excel"
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

# Limpeza dos nomes de colunas
df.columns = (
    df.columns
    .str.replace(r"[\n\r\xa0]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

st.caption(f"📄 Arquivo carregado: **{uploaded.name}** ({tipo_arquivo}) | Registros: **{len(df)}**")

# ===============================================================
# 🔑 ETAPA 2 — Configurar API Key
# ===============================================================
st.sidebar.header("🔑 Etapa 2 — Configurar OpenAI API Key")
api_key = st.sidebar.text_input("OpenAI API Key (opcional)", type="password", placeholder="sk-...")

if api_key:
    client = OpenAI(api_key=api_key)
    st.sidebar.success("✅ Chave carregada com sucesso.")
else:
    client = None
    st.sidebar.info("ℹ️ Recursos com GPT ficarão desativados até inserir a chave.")

st.sidebar.markdown("---")

# ===============================================================
# 🎯 ETAPA 3 — Aplicar Filtros Dinâmicos
# ===============================================================
st.sidebar.header("🎯 Etapa 3 — Aplicar Filtros")
filter_cols = ["Nome do curso", "Modalidade do curso", "Tipo Graduação", "Cluster", "Tipo do componente"]
df_filtered = df.copy()
active_filters = {}

# Filtros dependentes (respeitam seleção anterior)
for col in filter_cols:
    if col in df_filtered.columns:
        values = sorted(df_filtered[col].dropna().astype(str).unique())
        sel = st.sidebar.multiselect(col, values, default=[])
        if sel:
            df_filtered = df_filtered[df_filtered[col].astype(str).isin(sel)]
            active_filters[col] = sel

# ===============================================================
# 🧭 ETAPA 4 — Selecionar Tipo de Análise
# ===============================================================
st.sidebar.header("🧭 Etapa 4 — Selecionar Tipo de Análise")

menu = st.sidebar.selectbox(
    "Escolha a análise desejada:",
    [
        "1️⃣ 📊 Resumo Geral",
        "2️⃣ ✅ Cobertura Curricular",
        "3️⃣ 📈 Curva Bloom Progressiva",
        "4️⃣ 🎯 Alinhamento de Objetivos e Competências",
        "5️⃣ 🧩 Similaridade e Redundância",
        "6️⃣ 🌐 Convergência Temática",
        "7️⃣ 🔗 Dependência Curricular",
        "8️⃣ 💬 Clareza e Sentimento das Ementas",
        "9️⃣ 📆 Análise Longitudinal",
        "🔟 🤖 Relatório Consultivo",
    ],
    index=0
)

scope_key = normalize_text(menu).replace(" ", "_")
_init_exports(scope_key)

# ---------------------------------------------------------------
# 🔍 Exibir filtros aplicados
# ---------------------------------------------------------------
with st.expander("🔍 Filtros aplicados", expanded=False):
    if active_filters:
        for k, v in active_filters.items():
            st.write(f"**{k}:** {', '.join(map(str, v))}")
    else:
        st.caption("Nenhum filtro aplicado.")
st.markdown("---")

# ===============================================================
# 🚀 Execução da Análise Selecionada
# ===============================================================
if menu.endswith("📊 Resumo Geral") or "Resumo Geral" in menu:
    try:
        from modules.summary_dashboard import run_summary
        st.header("📊 Resumo Geral")
        st.caption("Visão geral dos dados importados, número de UCs, cursos e distribuição geral.")
        run_summary(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro ao carregar Resumo Geral: {e}")

elif "Cobertura Curricular" in menu:
    try:
        from modules.coverage_report import run_coverage
        st.header("✅ Cobertura Curricular")
        st.caption("Mapeia o grau de cobertura das competências e conteúdos previstos nas UCs.")
        run_coverage(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro na Cobertura Curricular: {e}")

elif "Curva Bloom Progressiva" in menu:
    try:
        from modules.bloom_analysis import run_bloom
        st.header("📈 Curva Bloom Progressiva")
        st.caption("Analisa o nível cognitivo predominante (Taxonomia de Bloom) dos objetivos de aprendizagem.")
        run_bloom(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro na Curva Bloom Progressiva: {e}")

elif "Alinhamento de Objetivos e Competências" in menu:
    try:
        from modules.alignment_topk import run_alignment
        st.header("🎯 Alinhamento de Objetivos e Competências")
        st.caption("Avalia a coerência entre os objetivos de aprendizagem e as competências do egresso.")
        run_alignment(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro no Alinhamento: {e}")

elif "Similaridade e Redundância" in menu:
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

elif "Convergência Temática" in menu:
    try:
        from modules.clusterization import run_cluster
        st.header("🌐 Convergência Temática")
        st.caption("Agrupa UCs por similaridade semântica, revelando convergências interdisciplinares.")
        run_cluster(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro na Convergência Temática: {e}")

elif "Dependência Curricular" in menu:
    try:
        from modules.dependency_graph import run_graph
        st.header("🔗 Dependência Curricular")
        st.caption("Identifica relações de precedência e interdependência entre UCs com base em inferência semântica.")
        run_graph(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro na Dependência Curricular: {e}")

elif "Clareza e Sentimento" in menu:
    try:
        from modules.sentiment_analysis import run_sentiment
        st.header("💬 Clareza e Sentimento das Ementas")
        st.caption("Analisa o tom e a clareza textual das ementas, detectando vieses ou falta de objetividade.")
        run_sentiment(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro em Clareza e Sentimento: {e}")

elif "Análise Longitudinal" in menu:
    try:
        from modules.longitudinal_analysis import run_longitudinal
        st.header("📆 Análise Longitudinal")
        st.caption("Acompanha revisões e evoluções curriculares ao longo dos semestres ou versões das ementas.")
        run_longitudinal(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro na Análise Longitudinal: {e}")

elif "Relatório Consultivo" in menu:
    try:
        from modules.consultive_report import run_consultive
        st.header("🤖 Relatório Consultivo")
        st.caption("Gera um relatório automatizado com diagnósticos e recomendações sobre a coerência curricular geral.")
        run_consultive(df_filtered, scope_key, client)
    except Exception as e:
        st.error(f"Erro no Relatório Consultivo: {e}")

# ===============================================================
# 📦 Exportação Global
# ===============================================================
st.markdown("---")
export_zip_button(scope_key)

# ===============================================================
# 🧭 Rodapé
# ===============================================================
st.markdown("---")
st.caption("""
📘 **EmentaLabv2** — Ferramenta de análise curricular inteligente.  
Desenvolvido para apoiar na revisão de coerência, progressão cognitiva e integração pedagógica.
""")
