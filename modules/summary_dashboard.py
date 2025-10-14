# ===============================================================
# 📊 EmentaLabv2 — Resumo Dashboard
# ===============================================================
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.exportkit import export_zip_button, show_and_export_fig
from utils.text_utils import find_col, _rotate_yticks

def run_summary(df, scope_key):
    """Resumo básico da base curricular filtrada"""
    st.header("📊 Resumo da Base Curricular")
    st.markdown("""
**O que é:** visão geral do escopo filtrado (quantidade de UCs, clusters e carga horária).  
**Como analisar:** confirme se o filtro selecionado é o desejado e observe a distribuição de tipos de componente.
    """)

    col_ch = find_col(df, "CH")
    col_tipo = find_col(df, "Tipo do componente")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de UCs", len(df))
    c2.metric("Clusters únicos", df["Cluster"].nunique() if "Cluster" in df.columns else "—")

    if col_ch and col_ch in df.columns:
        ch = pd.to_numeric(df[col_ch], errors="coerce")
        c3.metric("Carga horária total", f"{ch.sum(skipna=True):.0f}h")
    else:
        c3.metric("Carga horária total", "N/A")

    if col_tipo:
        freq = df[col_tipo].astype(str).value_counts().head(10)
        st.subheader("📊 Distribuição por Tipo do componente")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(y=freq.index, x=freq.values, ax=ax, palette="Blues_r")
        ax.set_xlabel("Quantidade de UCs"); ax.set_ylabel("")
        ax.set_title("Distribuição por Tipo do componente")
        _rotate_yticks(ax, size=9)
        show_and_export_fig(scope_key, fig, "resumo_tipo_componente")

    st.markdown("### 📦 Centro de Downloads (Resumo)")
    export_zip_button(scope_key)
