# ===============================================================
# 📈 EmentaLabv2 — Clusterização (Ementa)
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, show_and_export_fig, export_zip_button
from utils.text_utils import find_col, replace_semicolons

def run_cluster(df, scope_key):
    col_ementa = find_col(df, "Ementa")
    if not col_ementa:
        st.stop()
    df_an = df.dropna(subset=[col_ementa])
    textos = df_an[col_ementa].astype(str).apply(replace_semicolons).tolist()
    nomes = df_an["Nome da UC"].astype(str).tolist()
    emb = l2_normalize(sbert_embed(textos))

    k = st.slider("Número de clusters (K)", 2, 10, 5)
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(emb)

    df_out = pd.DataFrame({"Nome da UC": nomes, "Cluster": labels})
    st.dataframe(df_out)
    export_table(scope_key, df_out, "clusterizacao", "Clusters")
    export_zip_button(scope_key)
