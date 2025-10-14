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
# 🎨 Cabeçalho e Identidade
# ---------------------------------------------------------------
logo = Path("assets/logo.png")
if logo.exists():
    st.image(str(logo), width=250)

st.title("🧠 EmentaLabv2 — Inteligência Curricular")
st.markdown("Ferramenta integrada para análise e aprimoramento de matrizes curriculares.")
st.markdown("---")

# ---------------------------------------------------------------
# ETAPA 1️⃣ — Upload do Arquivo
# ---------------------------------------------------------------
st.header("📂 Etapa 1 — Carregar Base Curricular")

uploaded = st.file_uploader("Envie seu arquivo (.xlsx ou .csv)", type=["xlsx", "csv"])
if not uploaded:
    st.info("👆 Faça o upload da base curricular para continuar.")
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

st.success(f"✅ Base carregada com sucesso! ({len(df)} registros) — {sheet_info}")

st.markdown("---")

# ---------------------------------------------------------------
# ETAPA 2️⃣ — Inserir a Chave da API (opcional)
# ---------------------------------------------------------------
st.header("🔑 Etapa 2 — Configurar Chave da OpenAI (opcional)")
st.caption(
    "Algumas análises utilizam modelos de linguagem (GPT) para aprimorar resultados — "
    "como Bloom, Dependência Curricular e Relatório Consultivo."
)

api_key = st.text_input(
    "Informe sua chave OpenAI (formato: sk-...)", 
    type="password", 
    placeholder="sk-xxxxxxxxxxxxxxxx"
)

client = OpenAI(api_key=api_key) if api_key else None

if api_key:
    st.success("✅ Chave validada com sucesso.")
else:
    st.warning("⚠️ Nenhuma chave informada. Recursos com GPT ficarão desativados.")

st.markdown("---")

# ---------------------------------------------------------------
# ETAPA 3️⃣ — Filtros
# ---------------------------------------------------------------
st.header("🎯 Etapa 3 — Aplicar Filtros")
st.caption("Use os filtros abaixo para segmentar sua análise por curso, modalidade ou tipo de componente.")

filter_cols = ["Nome do curso", "Modalidade do curso", "Tipo Graduação", "Cluster", "Tipo do componente"]
df_filtered = df.copy()
active_filters = {}

cols = st.columns(2)
for i, col in enumerate(filter_cols):
    if col in df.columns:
        options = sorted(df[col].dropna().astype(str).unique())
        with cols[i % 2]:
            selected = st.multiselect(col, options, default=[])
            if selected:
                df_filtered = df_filtered[df_filtered[col].astype(str).isin(selected)]
                active_filters[col] = selected

with st.expander("🔍 Filtros aplicados", expanded=False):
    if active_filters:
        for k, v in active_filters.items():
            st.write(f"**{k}:** {', '.join(map(str, v))}")
    else:
        st.caption("Nenhum filtro aplicado. Todos os registros serão considerados.")

st.info(f"📊 Total de registros filtrados: **{len(df_filtered)}**")

st.markdown("---")

# ---------------------------------------------------------------
# ETAPA 4️⃣ — Seleção da Análise
# ---------------------------------------------------------------
st.header("📈 Etapa 4 — Escolher Tipo de Análise")
st.caption("Selecione o tipo de análise que deseja executar sobre a base curricular.")

menu = st.selectbox(
    "Escolha uma análise para executar:",
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

# inicializa escopo de exportação
scope_key = normalize_text(menu).replace(" ", "_")
_init_exports(scope_key)

st.markdown("---")

# ---------------------------------------------------------------
# 🚀 Execução da Análise Selecionada
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
