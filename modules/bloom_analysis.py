# ===============================================================
# 🧠 EmentaLabv2 — Mapa de Bloom (Heurística + GPT)
# ===============================================================
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from utils.text_utils import find_col
from utils.bloom_helpers import calculate_bloom_level
from utils.exportkit import export_table, show_and_export_fig, export_zip_button

def run_bloom(df, scope_key):
    col_obj = find_col(df, "Objetivo de aprendizagem")
    if not col_obj:
        st.warning("Coluna de objetivos não encontrada.")
        st.stop()

    df_out = calculate_bloom_level(df, col_obj)
    st.subheader("📊 Distribuição Heurística")
    freq = df_out["Nível Bloom Predominante"].value_counts(normalize=True).mul(100).round(1)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=freq.index, y=freq.values, ax=ax)
    ax.set_ylabel("% de UCs"); ax.set_xlabel("Nível de Bloom")
    ax.set_title("Distribuição Heurística de Bloom")
    show_and_export_fig(scope_key, fig, "bloom_distribuicao_heuristica")
    st.dataframe(df_out)
    export_table(scope_key, df_out, "bloom_tabela_heuristica", "Bloom")
    export_zip_button(scope_key)
