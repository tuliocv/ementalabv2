# ===============================================================
# 🧠 EmentaLabv2 — Inteligência Curricular (v11.1)
# ===============================================================
import streamlit as st
import pandas as pd
from pathlib import Path

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

# leitura de arquivo
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
api_key_input = st.sidebar.text_input("OpenAI API Key (opcional)", type="password", placeholder="sk-...")

# persiste a chave na sessão
if api_key_input:
    st.session_state["global_api_key"] = api_key_input

api_key = st.session_state.get("global_api_key", "")

if api_key:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        st.sidebar.success("✅ Chave carregada com sucesso.")
    except Exception as e:
        client = None
        st.sidebar.warning(f"⚠️ Não foi possível inicializar OpenAI: {e}")
else:
    client = None
    st.sidebar.info("ℹ️ Recursos com GPT ficarão desativados até inserir a chave.")

st.sidebar.markdown("---")

# ===============================================================
# 🎯 ETAPA 3 — Aplicar Filtros Dinâmicos (interdependentes)
# ===============================================================
st.sidebar.header("🎯 Etapa 3 — Aplicar Filtros")
df_filtered = df.copy()
active_filters = {}

# ordem canônica e interdependente
filter_cols = [
    "Nome do curso",
    "Modalidade do curso",
    "Tipo Graduação",
    "Cluster",
    "Tipo do componente",
]

for col in filter_cols:
    if col in df_filtered.columns:
        options = sorted(df_filtered[col].dropna().astype(str).unique())
        sel = st.sidebar.multiselect(col, options, default=[])
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
        "9️⃣ 📈 Mapa de Conectividade Curricular",
        "🔟 🤖 Relatório Consultivo",
    ],
    index=0
)

# define escopo e inicializa exportações
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
# 1) Resumo
if "Resumo Geral" in menu:
    try:
        from modules.summary_dashboard import run_summary
        st.header("📊 Resumo Geral")
        st.caption("Visão geral dos dados importados, número de UCs, cursos e distribuição geral.")
        run_summary(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro ao carregar Resumo Geral: {e}")

# 2) Cobertura Curricular
elif "Cobertura Curricular" in menu:
    try:
        from modules.coverage_report import run_coverage
        st.header("✅ Cobertura Curricular")
        st.caption("Mapeia o grau de cobertura das competências e conteúdos previstos nas UCs.")
        run_coverage(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro na Cobertura Curricular: {e}")

# 3) Curva Bloom
elif "Curva Bloom Progressiva" in menu:
    try:
        from modules.bloom_analysis import run_bloom
        st.header("📈 Curva Bloom Progressiva")
        st.caption("Analisa o nível cognitivo predominante (Taxonomia de Bloom) dos objetivos de aprendizagem.")
        # 👉 módulos recentes aceitam client opcional
        try:
            run_bloom(df_filtered, scope_key, client)
        except TypeError:
            run_bloom(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro na Curva Bloom Progressiva: {e}")

# 4) Alinhamento Objetivos × Competências
elif "Alinhamento de Objetivos e Competências" in menu:
    try:
        from modules.alignment_topk import run_alignment
        st.header("🎯 Alinhamento de Objetivos e Competências")
        st.caption("Avalia a coerência entre os objetivos de aprendizagem e as competências do egresso.")
        run_alignment(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro no Alinhamento: {e}")

# 5) Similaridade e Redundância (3 subanálises)
elif "Similaridade e Redundância" in menu:
    try:
        from modules.redundancy_matrix import run_redundancy, run_pair_analysis, run_alignment_matrix
        st.header("🧩 Similaridade e Redundância")
        st.caption("Detecta sobreposições, compara UCs frase a frase e mede o alinhamento Objetos × Competências & DCN.")
        tab1, tab2, tab3 = st.tabs([
            "🔁 Redundância entre UCs",
            "🔬 Comparação Frase a Frase",
            "🧭 Matriz Objetos × Competências & DCN"
        ])
        with tab1:
            run_redundancy(df_filtered, scope_key)
        with tab2:
            run_pair_analysis(df_filtered, scope_key)
        with tab3:
            # passa client se disponível
            try:
                run_alignment_matrix(df_filtered, scope_key, client)
            except TypeError:
                run_alignment_matrix(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro em Similaridade e Redundância: {e}")

# 6) Convergência Temática (Clusterização)
elif "Convergência Temática" in menu:
    try:
        # sua implementação pode estar em 'clusterization' ou 'cluster_analysis'
        try:
            from modules.clusterization import run_cluster
        except Exception:
            from modules.cluster_analysis import run_cluster
        st.header("🌐 Convergência Temática")
        st.caption("Agrupa UCs por similaridade semântica, revelando convergências interdisciplinares.")
        try:
            run_cluster(df_filtered, scope_key, client)
        except TypeError:
            run_cluster(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro na Convergência Temática: {e}")

# 7) Dependência Curricular (Grafo)
elif "Dependência Curricular" in menu:
    try:
        from modules.dependency_graph import run_graph
        st.header("🔗 Dependência Curricular")
        st.caption("Identifica relações de precedência e interdependência entre UCs com base em inferência semântica.")
        try:
            run_graph(df_filtered, scope_key, client)
        except TypeError:
            run_graph(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro na Dependência Curricular: {e}")

# 8) Clareza e Sentimento
elif "Clareza e Sentimento" in menu:
    try:
        from modules.sentiment_analysis import run_sentiment
        st.header("💬 Clareza e Sentimento das Ementas")
        st.caption("Analisa o tom e a clareza textual das ementas, detectando vieses ou falta de objetividade.")
        try:
            run_sentiment(df_filtered, scope_key, client)
        except TypeError:
            run_sentiment(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro em Clareza e Sentimento: {e}")

# 9) Mapa de Conectividade Curricular (nome do arquivo mantido: longitudinal_analysis.py)
elif "Mapa de Conectividade Curricular" in menu:
    try:
        from modules.longitudinal_analysis import run_longitudinal
        st.header("📈 Mapa de Conectividade Curricular")
        st.caption("Rede de impacto entre UCs via similaridade semântica e métricas de centralidade.")
        try:
            run_longitudinal(df_filtered, scope_key, client)
        except TypeError:
            run_longitudinal(df_filtered, scope_key)
    except Exception as e:
        st.error(f"Erro no Mapa de Conectividade Curricular: {e}")

# 10) Relatório Consultivo
elif "Relatório Consultivo" in menu:
    try:
        from modules.consultive_report import run_consultive
        st.header("🤖 Relatório Consultivo")
        st.caption("Gera um relatório automatizado com diagnósticos e recomendações sobre a coerência curricular geral.")
        try:
            run_consultive(df_filtered, scope_key, client)
        except TypeError:
            run_consultive(df_filtered, scope_key)
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
