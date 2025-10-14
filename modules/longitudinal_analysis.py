# ===============================================================
# 🌐 EmentaLabv2 — Mapa de Conectividade Curricular (v1.0)
# ===============================================================
# - Cria rede de impacto entre UCs via similaridade semântica
# - Calcula métricas de centralidade (grau, intermediação, densidade)
# - Identifica UCs estruturantes, periféricas e redundantes
# - Gera visualização de grafo e relatório analítico automático (GPT)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from openai import OpenAI

from utils.embeddings import sbert_embed, l2_normalize
from utils.text_utils import find_col, truncate
from utils.exportkit import export_table, export_zip_button, show_and_export_fig


# ---------------------------------------------------------------
# 🚀 Função principal
# ---------------------------------------------------------------
def run_connectivity(df, scope_key, client=None):
    st.header("🌐 Mapa de Conectividade Curricular (Rede de Impacto)")
    st.caption(
        """
        Analisa o **grau de conexão semântica entre as Unidades Curriculares (UCs)**, identificando
        disciplinas **estruturantes**, **intermediárias** e **periféricas** dentro da matriz curricular.
        Baseia-se em embeddings SBERT e métricas de rede.
        """
    )

    # -----------------------------------------------------------
    # 📂 Localiza colunas principais
    # -----------------------------------------------------------
    col_text = (
        find_col(df, "Ementa")
        or find_col(df, "Descrição")
        or find_col(df, "Objetos de conhecimento")
    )
    if not col_text:
        st.error("Coluna 'Ementa', 'Descrição' ou 'Objetos de conhecimento' não encontrada.")
        return

    col_uc = find_col(df, "Nome da UC")
    if not col_uc:
        st.error("Coluna 'Nome da UC' não encontrada.")
        return

    df_valid = df[[col_uc, col_text]].dropna().rename(columns={col_uc: "UC", col_text: "Texto"})
    if df_valid.empty:
        st.warning("Nenhuma UC com texto preenchido.")
        return

    max_uc = st.slider("Quantidade de UCs para análise", 4, min(50, len(df_valid)), min(20, len(df_valid)), 1)
    df_valid = df_valid.head(max_uc)

    # -----------------------------------------------------------
    # 🧠 Embeddings e matriz de similaridade
    # -----------------------------------------------------------
    with st.spinner("🧠 Calculando embeddings SBERT..."):
        textos = df_valid["Texto"].astype(str).tolist()
        emb = l2_normalize(sbert_embed(textos))
        sims = np.dot(emb, emb.T)

    df_sim = pd.DataFrame(sims, index=df_valid["UC"], columns=df_valid["UC"])
    export_table(scope_key, df_sim, "matriz_similaridade", "Matriz de Similaridade entre UCs")

    st.markdown("### 🔍 Mapa de Similaridade Semântica")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_sim, cmap="crest", linewidths=0.5)
    ax.set_title("Matriz de Similaridade entre UCs (SBERT)", fontsize=12)
    st.pyplot(fig, use_container_width=True)
    show_and_export_fig(scope_key, fig, "mapa_similaridade_semantica")

    # -----------------------------------------------------------
    # 🕸️ Criação do grafo de conectividade
    # -----------------------------------------------------------
    st.markdown("### 🕸️ Rede de Impacto Curricular")
    threshold = st.slider("Limite de conexão (similaridade mínima)", 0.5, 0.95, 0.75, 0.05)

    G = nx.Graph()
    for i, uc_a in enumerate(df_valid["UC"]):
        for j, uc_b in enumerate(df_valid["UC"]):
            if i < j and sims[i, j] >= threshold:
                G.add_edge(uc_a, uc_b, weight=sims[i, j])

    if G.number_of_edges() == 0:
        st.warning("Nenhuma conexão encontrada com o limite selecionado. Reduza o threshold.")
        return

    # -----------------------------------------------------------
    # 📊 Métricas de centralidade
    # -----------------------------------------------------------
    centralidade_grau = nx.degree_centrality(G)
    centralidade_inter = nx.betweenness_centrality(G)
    densidade = nx.density(G)

    df_centralidade = pd.DataFrame({
        "UC": list(G.nodes),
        "Centralidade Grau": [centralidade_grau[n] for n in G.nodes],
        "Centralidade Intermediação": [centralidade_inter[n] for n in G.nodes],
    }).sort_values("Centralidade Grau", ascending=False)

    st.markdown("### 📈 Disciplinas Estruturantes (Maior Centralidade)")
    st.dataframe(df_centralidade, use_container_width=True)
    export_table(scope_key, df_centralidade, "centralidade_uc", "Centralidade das UCs na Rede de Impacto")

    # -----------------------------------------------------------
    # 🎨 Visualização do grafo
    # -----------------------------------------------------------
    st.markdown("### 🎨 Visualização da Rede")
    pos = nx.spring_layout(G, seed=42, k=0.6)
    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#90CAF9", edgecolors="#1565C0")
    nx.draw_networkx_edges(G, pos, width=1.8, alpha=0.7, edge_color="#42A5F5")
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Mapa de Conectividade Curricular", fontsize=13, fontweight="bold")
    plt.axis("off")
    st.pyplot(fig, use_container_width=True)
    show_and_export_fig(scope_key, fig, "grafo_conectividade_curricular")

    # -----------------------------------------------------------
    # 🧠 Análise automática via GPT (opcional)
    # -----------------------------------------------------------
    if client is None:
        api_key = st.session_state.get("global_api_key", "")
        if api_key:
            client = OpenAI(api_key=api_key)

    if client is not None:
        top_nodes = df_centralidade.head(5)["UC"].tolist()
        bottom_nodes = df_centralidade.tail(5)["UC"].tolist()

        resumo = (
            f"Foram analisadas {len(G.nodes)} UCs, com densidade média de {densidade:.2f}. "
            f"As UCs mais conectadas (estruturantes) são: {', '.join(top_nodes)}. "
            f"As menos conectadas (periféricas) são: {', '.join(bottom_nodes)}."
        )

        prompt = (
            "Você é um avaliador curricular. Analise o resumo abaixo e produza um relatório curto e técnico, "
            "com foco em **pontos fortes**, **fragilidades** e **recomendações práticas** para melhoria da "
            "estrutura curricular com base na rede de conectividade.\n\n"
            f"{resumo}"
        )

        try:
            with st.spinner("📄 Gerando relatório analítico via GPT..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            analise = resp.choices[0].message.content.strip()
            st.markdown("### 🧾 Relatório Analítico da Estrutura Curricular")
            st.info(analise)
        except Exception as e:
            st.error(f"❌ Erro ao gerar relatório analítico via GPT: {e}")

    # -----------------------------------------------------------
    # 🧭 Interpretação e aplicação
    # -----------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        ## 🧭 Como interpretar os resultados
        - **Alta centralidade de grau:** UC muito conectada → conteúdo fundamental, amplamente relacionado.  
        - **Alta intermediação:** UC que atua como ponte entre áreas → interdisciplinar.  
        - **Baixa centralidade:** UC isolada ou muito específica → pode indicar especialização ou desconexão temática.  
        - **Alta densidade da rede:** curso coeso e bem articulado.  
        - **Baixa densidade:** curso fragmentado, com possíveis lacunas entre áreas.  

        ### 🧩 Aplicações práticas
        - Identificar **disciplinas estruturantes** (núcleo formativo).  
        - Detectar **redundâncias e sobreposição** temática.  
        - Mapear **áreas que necessitam maior integração**.  
        - Apoiar revisões de **matriz curricular e PPC** com evidências gráficas.
        """
    )

    export_zip_button(scope_key)
