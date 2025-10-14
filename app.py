# ===============================================================
# 🧭 EmentaLabv2 — Versão 10.0 (com menu lateral de API Key)
# ===============================================================

import streamlit as st
import pandas as pd
from openai import OpenAI
from utils.exportkit import _init_exports

# ---------------------------------------------------------------
# 🧱 Configuração da Página
# ---------------------------------------------------------------
st.set_page_config(
    page_title="EmentaLabv2 — Análise Curricular Inteligente",
    page_icon="🧭",
    layout="wide"
)

# ---------------------------------------------------------------
# 🎨 Sidebar — Menu Principal
# ---------------------------------------------------------------
st.sidebar.image("assets/logo.png", width=220)
st.sidebar.markdown("---")
st.sidebar.header("🔍 Menu de Análises")

menu = st.sidebar.radio(
    "Selecione o tipo de análise:",
    [
        "📈 Curva Bloom Progressiva",
        "🎯 Alinhamento Objetivos × Competências",
        "🧬 Redundância e Frase-a-Frase",
        "📊 Clusterização Temática",
        "🔗 Dependência Curricular (Grafo)",
    ],
    key="menu_principal"
)

st.sidebar.markdown("---")

# ---------------------------------------------------------------
# 🔑 Configuração Global da API OpenAI
# ---------------------------------------------------------------
st.sidebar.markdown("## 🔑 Configurações Globais")
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Informe sua chave da OpenAI. Ela será usada apenas localmente nesta sessão."
)

client = OpenAI(api_key=api_key) if api_key else None

if api_key:
    st.sidebar.success("✅ Chave carregada com sucesso.")
else:
    st.sidebar.info("ℹ️ Insira a API Key para habilitar recursos GPT.")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Dica: a chave é usada em módulos que utilizam GPT (Bloom, Alinhamento, Dependência).")

# ---------------------------------------------------------------
# 📤 Upload de Arquivo
# ---------------------------------------------------------------
st.sidebar.header("📂 Importar Dados")
uploaded = st.sidebar.file_uploader(
    "Envie sua planilha (Excel ou CSV)",
    type=["xlsx", "csv"]
)

if uploaded is not None:
    try:
        df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        st.stop()
else:
    st.warning("Envie um arquivo para começar a análise.")
    st.stop()

# ---------------------------------------------------------------
# 🧾 Inicializa pasta de exportação temporária
# ---------------------------------------------------------------
scope_key = _init_exports()

# ---------------------------------------------------------------
# 🚀 Roteamento por Módulo
# ---------------------------------------------------------------
if menu == "📈 Curva Bloom Progressiva":
    from modules.bloom_analysis import run_bloom
    st.title("📈 Curva Bloom Progressiva")
    st.markdown("""
    Esta análise avalia o **nível cognitivo predominante** dos objetivos de aprendizagem,
    utilizando heurísticas linguísticas e (opcionalmente) o modelo GPT para refinar a classificação segundo a Taxonomia de Bloom.
    """)
    run_bloom(df, scope_key, client)

elif menu == "🎯 Alinhamento Objetivos × Competências":
    from modules.alignment_analysis import run_alignment
    st.title("🎯 Alinhamento Objetivos × Competências")
    st.markdown("""
    Analisa o **grau de coerência** entre os **objetivos de aprendizagem** e as **competências do perfil do egresso**,
    utilizando similaridade semântica baseada em embeddings SBERT.
    """)
    run_alignment(df, scope_key)

elif menu == "🧬 Redundância e Frase-a-Frase":
    from modules.redundancy_analysis import run_redundancy, run_pair_analysis
    st.title("🧬 Redundância e Análise Frase-a-Frase")
    st.markdown("""
    Esta ferramenta identifica **sobreposições de conteúdo** entre UCs e permite comparar **trechos de ementas** frase a frase.
    Ideal para revisar redundâncias e ajustar coerência curricular.
    """)
    tab1, tab2 = st.tabs(["🔁 Redundância entre UCs", "🔬 Comparação Frase-a-Frase"])
    with tab1:
        run_redundancy(df, scope_key)
    with tab2:
        run_pair_analysis(df, scope_key)

elif menu == "📊 Clusterização Temática":
    from modules.clusterization import run_cluster
    st.title("📊 Clusterização Temática das Ementas")
    st.markdown("""
    Agrupa as Unidades Curriculares (UCs) por **proximidade semântica dos conteúdos**.
    Pode usar o GPT para **nomear automaticamente os clusters** e gerar visualizações comparativas.
    """)
    run_cluster(df, scope_key, client)

elif menu == "🔗 Dependência Curricular (Grafo)":
    from modules.dependency_graph_interactive import run_graph_interactive
    st.title("🔗 Mapa de Dependências Curriculares")
    st.markdown("""
    Este módulo identifica **relações de precedência e interdependência** entre as UCs,
    gerando um **mapa hierárquico estático** com base nas ementas e justificativas textuais do GPT.
    """)
    run_graph_interactive(df, scope_key, client)

# ---------------------------------------------------------------
# 🧭 Rodapé
# ---------------------------------------------------------------
st.markdown("---")
st.caption("""
📘 **EmentaLabv2** — Ferramenta de análise curricular inteligente.
Desenvolvido para apoiar **NDEs e coordenações** na revisão de coerência e integração pedagógica entre Unidades Curriculares.
""")
