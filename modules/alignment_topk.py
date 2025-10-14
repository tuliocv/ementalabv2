# ===============================================================
# 🎯 EmentaLabv2 — Alinhamento Objetivos × Competências
# ===============================================================
import streamlit as st
import numpy as np
import pandas as pd
from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, export_zip_button
from utils.text_utils import find_col, replace_semicolons

def run_alignment(df, scope_key):
    col_obj = find_col(df, "Objetivo de aprendizagem")
    col_comp = find_col(df, "Competências do Perfil do Egresso")
    if not col_obj or not col_comp:
        st.error("Faltam colunas para o alinhamento.")
        st.stop()

    df_an = df[[ "Nome da UC", col_obj, col_comp ]].dropna()
    objs = df_an[col_obj].apply(replace_semicolons).tolist()
    comps = df_an[col_comp].apply(replace_semicolons).tolist()
    nomes = df_an["Nome da UC"].tolist()

    emb_A = l2_normalize(sbert_embed(objs))
    emb_B = l2_normalize(sbert_embed(comps))
    S = np.dot(emb_A, emb_B.T)

    k = st.slider("Top-k (competências mais alinhadas por UC)", 3, 15, 5)
    excluir = st.checkbox("Excluir a própria UC", True)

    rows = []
    for i in range(S.shape[0]):
        sims = S[i].copy()
        if excluir: sims[i] = -1
        idx = np.argsort(-sims)[:k]
        for rank, j in enumerate(idx, start=1):
            rows.append({"UC (Objetivo)": nomes[i], "UC (Competência)": nomes[j], "Rank": rank, "Similaridade": float(S[i,j])})

    df_out = pd.DataFrame(rows)
    st.dataframe(df_out.head(1000), use_container_width=True)
    export_table(scope_key, df_out, "alinhamento_topk", "Alinhamento")
    export_zip_button(scope_key)
