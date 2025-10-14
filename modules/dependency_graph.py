import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_dep_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def dependency_graph(df, col_ementa="EMENTA", col_nome="NOME UC"):
    st.header("🔗 Análise de Dependência Curricular (Grafo Dirigido)")

    if col_ementa not in df.columns or col_nome not in df.columns:
        st.warning("A base deve conter as colunas 'NOME UC' e 'EMENTA'.")
        return

    model = load_dep_model()
    emb = model.encode(df[col_ementa].tolist(), convert_to_tensor=True)
    sim = util.cos_sim(emb, emb).cpu().numpy()

    G = nx.DiGraph()
    for i, uc1 in enumerate(df[col_nome]):
        for j, uc2 in enumerate(df[col_nome]):
            if i != j and sim[i, j] > 0.8:
                G.add_edge(uc1, uc2, weight=sim[i, j])

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(9, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_size=8)
    st.pyplot(plt)
    st.info(f"Grafo com {len(G.nodes)} nós e {len(G.edges)} relações detectadas.")
    return G
