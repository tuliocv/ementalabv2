import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

bloom_order = {"Lembrar":1, "Compreender":2, "Aplicar":3, "Analisar":4, "Avaliar":5, "Criar":6}

def bloom_progressive(df):
    st.header("📈 Curva de Complexidade Cognitiva (Bloom Progressivo)")

    if "BLOOM" not in df.columns or "PERIODO" not in df.columns:
        st.warning("A base deve conter as colunas 'BLOOM' e 'PERIODO'.")
        return

    df["BloomNum"] = df["BLOOM"].map(bloom_order)
    curva = df.groupby("PERIODO")["BloomNum"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=curva, x="PERIODO", y="BloomNum", marker="o", ax=ax, color="#006699")
    ax.set_title("Evolução Cognitiva por Período")
    ax.set_ylabel("Nível Médio (Bloom)")
    ax.set_xlabel("Período")
    st.pyplot(fig)
    st.success("Curva de Bloom gerada com sucesso.")
    return curva
