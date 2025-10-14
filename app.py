# ===============================================================
# 🧠 EmentaLabv2 — Inteligência Curricular (v10.4)
# ===============================================================
import pandas as pd
import streamlit as st
from pathlib import Path

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

# 🔑 API key global (usada por módulos que utilizam GPT)
st.sidebar.subheader("🔑 Configurações")
api_key = st.sidebar.text_input(
    "OpenAI API Key (opcional)",
    type="password",
    placeholder="sk-...",
    help="Se informada, será usada nos módulos que utilizam GPT (nome de clusters, Bloom, grafo, relatório etc.)"
)
st.sidebar.markdown("---")

# ---------------------------------------------------------------
# 📂 Upload de Arquivo
# ---------------------------------------------------------------
st.sidebar.header("📂 Base Curricular")
uploaded = st.sidebar.file_uploader("Carregar arquivo (.xlsx ou .csv)", type=["xlsx", "csv"])
if not uploaded:
    st.info("👈 Envie um arquivo Excel ou CSV para iniciar.")
    st.stop()

# Leitura resiliente
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

# Higiene de colunas
df.columns = (
    df.columns
      .str.replace(r"[\n\r\xa0]", " ", regex=True)
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
)

st.caption(f"📄 Arquivo: **{uploaded.name}** {sheet_info} | Registros: **{len(df)}**")

# ---------------------------------------------------------------
# 🎯 Filtros Essenciais (sempre visíveis)
# ---------------------------------------------------------------
st.sidebar.subheader("🎯 Filtros")
filter_cols = ["Nome do curso", "Modalidade do curso", "Tipo Graduação", "Cluster", "Tipo do componente"]
active_filters = {}
df_filtered = df.copy()

for col in filter_cols:
    if col in df_filtered.columns:
        values = sorted(df_filtered[col].dropna().astype(str).unique())
        sel = st.sidebar.multiselect(col, values, default=[])
        if sel:
            df_filtered = df_filtered[df_filtered[col].astype(str).isin(sel)]
            active_filters[col] = sel

# ---------------------------------------------------------------
# 🧭 Menu de Análises (apenas nomes institucionais)
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

# chave de escopo para exportação
scope_key = normalize_text(menu).replace(" ", "_")
_init_exports(scope_key)

# ---------------------------------------------------------------
# 🔎 Cabeçalho comum (contexto do filtro)
# ---------------------------------------------------------------
with st.expander("🔎 Contexto do filtro aplicado", expanded=False):
    if active_filters:
        for k, v in active_filters.items():
            st.write(f"**{k}:** {', '.join(map(str, v))}")
    else:
        st.caption("Nenhum filtro aplicado.")

st.markdown("---")

# ---------------------------------------------------------------
# 🚀 Roteamento por Análise
# ---------------------------------------------------------------
if menu == "📊 Resumo Geral":
    # Resumo rápido inline para garantir que sempre exista uma visão inicial
    try:
        from modules.summary_dashboard import run_summary
        run_summary(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Falha ao abrir Resumo Geral: {e}")

elif menu == "✅ Cobertura Curricular":
    try:
        from modules.coverage_report import run_coverage
        run_coverage(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Falha ao abrir Cobertura Curricular: {e}")

elif menu == "📈 Curva Bloom Progressiva":
    try:
        from modules.bloom_analysis import run_bloom
        run_bloom(df_filtered, scope_key, api_key=api_key)
    except Exception as e:
        st.error(f"Falha na Curva Bloom Progressiva: {e}")

elif menu == "🎯 Alinhamento de Objetivos e Competências":
    try:
        from modules.alignment_topk import run_alignment
        run_alignment(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Falha no Alinhamento: {e}")

elif menu == "🧩 Similaridade e Redundância":
    try:
        from modules.redundancy_matrix import run_redundancy
        run_redundancy(df_filtered, scope_key)
        st.markdown("---")
        from modules.redundancy_matrix import run_pair_analysis
        run_pair_analysis(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Falha em Similaridade/Redundância: {e}")

elif menu == "🌐 Convergência Temática":
    try:
        from modules.clusterization import run_cluster
        run_cluster(df_filtered, scope_key, api_key=api_key)
    except Exception as e:
        st.error(f"Falha na Convergência Temática: {e}")

elif menu == "🔗 Dependência Curricular":
    try:
        # Caso prefira versão estática organizada (sem PyVis), deixe apenas dependency_graph
        from modules.dependency_graph import run_graph
        run_graph(df_filtered, scope_key, api_key=api_key)
        # Se quiser a versão interativa e tiver dependências instaladas, troque pela linha abaixo:
        # from modules.dependency_graph_interactive import run_graph_interactive
        # run_graph_interactive(df_filtered, scope_key, api_key=api_key)
    except Exception as e:
        st.error(f"Falha em Dependência Curricular: {e}")

elif menu == "💬 Clareza e Sentimento das Ementas":
    try:
        from modules.sentiment_analysis import run_sentiment
        run_sentiment(df_filtered, scope_key, api_key=api_key)
    except Exception as e:
        st.error(f"Falha em Clareza e Sentimento: {e}")

elif menu == "📆 Análise Longitudinal":
    try:
        from modules.longitudinal_analysis import run_longitudinal
        run_longitudinal(df_filtered, scope_key, api_key=api_key)
    except Exception as e:
        st.error(f"Falha na Análise Longitudinal: {e}")

elif menu == "🤖 Relatório Consultivo":
    try:
        from modules.consultive_report import run_consultive
        run_consultive(df_filtered, scope_key, api_key=api_key)
    except Exception as e:
        st.error(f"Falha no Relatório Consultivo: {e}")

# ---------------------------------------------------------------
# 📦 Exportação (escopo atual)
# ---------------------------------------------------------------
st.markdown("---")
export_zip_button(scope_key)

# ---------------------------------------------------------------
# 🧭 Rodapé
# ---------------------------------------------------------------
st.markdown("---")
st.caption(
    "📘 EmentaLabv2 — análise curricular inteligente para NDEs e coordenações. "
    "Foca em coerência, progressão cognitiva, integração pedagógica e governança de revisões."
)
