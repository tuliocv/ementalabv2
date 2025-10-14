# ===============================================================
# 🌐 EmentaLabv2 — Mapa de Conectividade Curricular (v1.3)
# ===============================================================
# - Remove duplicações visuais (heatmap e grafo)
# - Gera matriz de similaridade + grafo únicos
# - Produz relatório GPT breve e direto
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from openai import OpenAI

from utils.embeddings import sbert_embed, l2_normalize
from utils.text_utils import find_col
from utils.exportkit import export_table, export_zip_button


# ---------------------------------------------------------------
# 🚀 Função principal
# ---------------------------------------------------------------
def run_longitudinal(df, scope_key, client=None):
    st.header("🌐 Mapa de Conectividade Curricular (Rede de Impacto)")
    st.caption(
        """
        Analisa a **conectividade semântica entre as Unidades Curriculares (UCs)**, destacando 
        disciplinas **estruturantes**, **intermediárias** e **periféricas** com base na similaridade 
        de suas ementas e objetos de conhecimento.
        """
    )

    # -----------------------------------------------------------
    # 📂 Identificação das colunas relevantes
    # -----------------------------------------------------------
    col_text = (
        find_col(df, "Ementa")
        or find_col(df, "Descrição")
        or find_col(df, "Objetos de conhecimento")
    )
    col_uc = find_col(df, "Nome da UC")

    if not col_uc or not col_text:
        st.error("Colunas 'Nome da UC' e 'Ementa' (ou equivalente) são necessárias.")
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
        emb = l2_normalize(sbert_embed(df_valid["Texto"].astype(str).tolist()))
        sims = np.dot(emb, emb.T)

    df_sim = pd.DataFrame(sims, index=df_valid["UC"], columns=df_valid["UC"])
    export_table(scope_key, df_sim, "matriz_similaridade", "Matriz de Similaridade entre UCs")

    # -----------------------------------------------------------
    # 🔍 Exibição única — Matriz de Similaridade
    # -----------------------------------------------------------
    with st.expander("📊 Matriz de Similaridade entre UCs", expanded=True):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df_sim, cmap="crest", linewidths=0.4)
        ax.set_title("Similaridade Semântica entre UCs (SBERT)", fontsize=11)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # -----------------------------------------------------------
    # 🕸️ Construção da Rede de Conectividade
    # -----------------------------------------------------------
    st.markdown("### 🕸️ Mapa de Conectividade Curricular")
    threshold = st.slider("Limite de conexão (similaridade mínima)", 0.5, 0.95, 0.75, 0.05)

    G = nx.Graph([
        (a, b, {"weight": sims[i, j]})
        for i, a in enumerate(df_valid["UC"])
        for j, b in enumerate(df_valid["UC"])
        if i < j and sims[i, j] >= threshold
    ])

    if G.number_of_edges() == 0:
        st.warning("Nenhuma conexão encontrada com o limite atual. Reduza o threshold.")
        return

    # -----------------------------------------------------------
    # 📊 Métricas de centralidade
    # -----------------------------------------------------------
    grau = nx.degree_centrality(G)
    inter = nx.betweenness_centrality(G)
    densidade = nx.density(G)

    df_centralidade = (
        pd.DataFrame({
            "UC": list(G.nodes),
            "Centralidade Grau": [grau[n] for n in G.nodes],
            "Centralidade Intermediação": [inter[n] for n in G.nodes],
        })
        .sort_values("Centralidade Grau", ascending=False)
    )

    with st.expander("📈 Centralidade das Disciplinas", expanded=True):
        st.dataframe(df_centralidade, use_container_width=True)
        export_table(scope_key, df_centralidade, "centralidade_uc", "Centralidade das UCs")

    # -----------------------------------------------------------
    # 🎨 Visualização única do Grafo
    # -----------------------------------------------------------
    st.markdown("### 🎨 Rede de Impacto Curricular")
    pos = nx.spring_layout(G, seed=42, k=0.6)
    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="#A5D8FF", edgecolors="#1565C0")
    nx.draw_networkx_edges(G, pos, width=1.4, alpha=0.75, edge_color="#64B5F6")
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Mapa de Conectividade Curricular", fontsize=12, fontweight="bold")
    plt.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # -----------------------------------------------------------
    # 🧠 Relatório Analítico via GPT
    # -----------------------------------------------------------
    api_key = st.session_state.get("global_api_key", "") if client is None else None
    if api_key:
        client = OpenAI(api_key=api_key)

    if client is not None:
        top_uc = df_centralidade.head(5)["UC"].tolist()
        low_uc = df_centralidade.tail(5)["UC"].tolist()

        resumo = (
            f"Foram analisadas {len(G.nodes)} UCs com densidade média de {densidade:.2f}. "
            f"As UCs mais conectadas (estruturantes) são: {', '.join(top_uc)}. "
            f"As menos conectadas (periféricas) são: {', '.join(low_uc)}."
        )

        prompt = (
            "Você é um avaliador curricular. Com base no resumo abaixo, produza um relatório breve e direto, "
            "indicando **pontos fortes**, **fragilidades** e **sugestões práticas** para melhoria da "
            "estrutura curricular:\n\n"
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
            st.error(f"❌ Erro ao gerar relatório via GPT: {e}")

    # -----------------------------------------------------------
    # 🧭 Interpretação final (concisa e não repetida)
    # -----------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        ### 🧭 Interpretação dos Resultados
        - **UCs estruturantes:** alta centralidade → sustentam o eixo formativo principal.  
        - **UCs intermediárias:** conectam diferentes áreas → papel integrador.  
        - **UCs periféricas:** baixa centralidade → podem indicar isolamento temático.  
        - **Alta densidade:** curso coeso e articulado.  
        - **Baixa densidade:** curso fragmentado, com lacunas entre eixos.

        🔹 **Aplicação prática:** identificar disciplinas centrais, revisar sobreposições e 
        promover maior integração entre eixos temáticos.
        """
    )

    export_zip_button(scope_key)
