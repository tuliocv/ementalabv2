# ===============================================================
# 🧩 EmentaLabv2 — Similaridade Objetos × Comp & DCN
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, export_zip_button
from utils.text_utils import find_col, replace_semicolons

def run_similarity(df, scope_key):
    col_obj = find_col(df, "Objetos de conhecimento")
    col_comp = find_col(df, "Competências do Perfil do Egresso")
    col_dcn = find_col(df, "Relação competência DCN")

    if not all([col_obj, col_comp, col_dcn]):
        st.error("Faltam colunas necessárias para a análise de similaridade.")
        st.stop()

    df_an = df[[ "Nome da UC", col_obj, col_comp, col_dcn ]].dropna(subset=[col_obj])
    st.header("🧩 Similaridade (Objetos × Competências & DCN)")
    st.caption("Valores próximos de 1 indicam forte alinhamento semântico entre Objetos e Competências/DCN.")

    objetos = df_an[col_obj].astype(str).apply(replace_semicolons).tolist()
    comp = df_an[col_comp].astype(str).tolist()
    dcn = df_an[col_dcn].astype(str).tolist()
    nomes = df_an["Nome da UC"].astype(str).tolist()

    with st.spinner("Gerando embeddings SBERT..."):
        emb_obj = l2_normalize(sbert_embed(objetos))
        emb_comp = l2_normalize(sbert_embed(comp))
        emb_dcn = l2_normalize(sbert_embed(dcn))

    df_sim = pd.DataFrame({
        "Nome da UC": nomes,
        "Sim. Objetos × Competências": np.sum(emb_obj * emb_comp, axis=1),
        "Sim. Objetos × Relação DCN": np.sum(emb_obj * emb_dcn, axis=1)
    })
    st.dataframe(df_sim.style.format("{:.3f}").background_gradient(cmap="RdYlGn", subset=["Sim. Objetos × Competências","Sim. Objetos × Relação DCN"]))
    export_table(scope_key, df_sim, "similaridade_obj_comp_dcn", "Similaridade")
    export_zip_button(scope_key)
