# ===============================================================
# 🧬 EmentaLabv2 — Redundância e Análise Frase-a-Frase
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, show_and_export_fig, export_zip_button
from utils.text_utils import find_col, replace_semicolons, _rotate_xticks, _rotate_yticks, _split_sentences

def run_redundancy(df, scope_key):
    col_base = find_col(df, "Ementa") or find_col(df, "Objetos de conhecimento")
    if not col_base:
        st.error("Coluna de texto não encontrada.")
        st.stop()

    textos = df[col_base].astype(str).apply(replace_semicolons).tolist()
    nomes = df["Nome da UC"].astype(str).tolist()
    emb = l2_normalize(sbert_embed(textos))
    S = np.dot(emb, emb.T)

    st.header("🧬 Redundância entre UCs")
    st.caption("Valores altos indicam conteúdos muito similares (redundância).")

    df_mat = pd.DataFrame(S, index=nomes, columns=nomes)
    st.dataframe(df_mat.head(30).style.format("{:.2f}").background_gradient(cmap="RdYlGn_r", vmin=0, vmax=1))
    export_table(scope_key, df_mat, "redundancia_matriz", "Redundância")

    thr = st.slider("Limiar de redundância", 0.5, 0.95, 0.8)
    pares = []
    n = S.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if S[i,j] >= thr:
                pares.append({"UC A": nomes[i], "UC B": nomes[j], "Similaridade": float(S[i,j])})
    df_pares = pd.DataFrame(pares)
    st.dataframe(df_pares.head(100))
    export_table(scope_key, df_pares, "redundancia_pares", "Pares")
    export_zip_button(scope_key)

def run_pair_analysis(df, scope_key):
    """Comparação frase a frase entre duas UCs"""
    st.header("🔬 Análise Frase a Frase")
    col_base = find_col(df, "Ementa") or find_col(df, "Objetos de conhecimento")
    if not col_base:
        st.stop()

    nomes = df["Nome da UC"].dropna().unique().tolist()
    uc_a = st.selectbox("UC A", nomes)
    uc_b = st.selectbox("UC B", [n for n in nomes if n != uc_a])

    text_a = replace_semicolons(df.loc[df["Nome da UC"]==uc_a, col_base].iloc[0])
    text_b = replace_semicolons(df.loc[df["Nome da UC"]==uc_b, col_base].iloc[0])
    ph_a, ph_b = _split_sentences(text_a), _split_sentences(text_b)
    emb_a, emb_b = sbert_embed(ph_a), sbert_embed(ph_b)
    sim = cosine_similarity(emb_a, emb_b)

    rows = []
    for i in range(len(ph_a)):
        j = np.argmax(sim[i])
        rows.append({"Similaridade": sim[i,j], "Trecho A": ph_a[i], "Trecho B": ph_b[j]})
    df_out = pd.DataFrame(rows).sort_values("Similaridade", ascending=False)
    st.dataframe(df_out.head(15).style.format({"Similaridade":"{:.3f}"}))
    export_table(scope_key, df_out, f"redundancia_{uc_a}_vs_{uc_b}", "Redundância Frases")
