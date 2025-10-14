import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
import streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def thematic_convergence(df, col_cluster="CLUSTER", col_ementa="EMENTA"):
    st.header("🌐 Mapa de Convergência Temática (Interdisciplinaridade)")

    if col_cluster not in df.columns or col_ementa not in df.columns:
        st.warning("A base deve conter as colunas 'CLUSTER' e 'EMENTA'.")
        return

    model = load_model()
    emb = model.encode(df[col_ementa].tolist(), convert_to_tensor=True)
    sim = util.cos_sim(emb, emb).cpu().numpy()
    df["Convergência"] = np.mean(sim, axis=1)

    cluster_mean = df.groupby(col_cluster)["Convergência"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=cluster_mean, x=col_cluster, y="Convergência", ax=ax, color="#009688")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    top_ucs = df.nlargest(5, "Convergência")[[col_ementa, "CLUSTER", "Convergência"]]
    st.dataframe(top_ucs, use_container_width=True)
    st.success("Análise de convergência temática concluída.")
    return cluster_mean
