# ===============================================================
# 🎯 EmentaLabv2 — Alinhamento Objetivos × Competências (v9.2)
# ===============================================================
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, export_zip_button
from utils.text_utils import find_col, replace_semicolons

def run_alignment(df, scope_key):
    # -----------------------------------------------------------
    # 🏷️ Título e descrição da análise
    # -----------------------------------------------------------
    st.header("🎯 Alinhamento (Objetivos × Competências)")
    st.caption(
        """
        Esta análise avalia o **grau de coerência entre os Objetivos de Aprendizagem e as Competências**
        descritas nas Unidades Curriculares (UCs).  
        É calculada a similaridade semântica entre os textos utilizando embeddings (SBERT),
        permitindo identificar **quanto os objetivos realmente refletem as competências declaradas**.
        """
    )

    # -----------------------------------------------------------
    # 📂 Localização das colunas
    # -----------------------------------------------------------
    col_obj = find_col(df, "Objetivo de aprendizagem")
    col_comp = find_col(df, "Competências do Perfil do Egresso")
    if not col_obj or not col_comp:
        st.error("Faltam colunas obrigatórias: 'Objetivo de aprendizagem' e/ou 'Competências do Perfil do Egresso'.")
        st.stop()

    df_an = df[["Nome da UC", col_obj, col_comp]].dropna()
    if df_an.empty:
        st.warning("Nenhuma UC possui ambos os campos preenchidos.")
        st.stop()

    # -----------------------------------------------------------
    # ⚙️ Parâmetros de análise
    # -----------------------------------------------------------
    k = st.slider("Top-k (competências mais alinhadas por UC)", 3, 15, 5)
    excluir = st.checkbox("Excluir a própria UC da comparação", True)

    # -----------------------------------------------------------
    # 🔢 Embeddings e similaridade
    # -----------------------------------------------------------
    st.info("🧠 Calculando embeddings SBERT e matriz de similaridade semântica...")
    objs = df_an[col_obj].apply(replace_semicolons).tolist()
    comps = df_an[col_comp].apply(replace_semicolons).tolist()
    nomes = df_an["Nome da UC"].tolist()

    emb_A = l2_normalize(sbert_embed(objs))
    emb_B = l2_normalize(sbert_embed(comps))
    S = np.dot(emb_A, emb_B.T)

    # -----------------------------------------------------------
    # 📈 Construção da tabela Top-k
    # -----------------------------------------------------------
    rows = []
    for i in range(S.shape[0]):
        sims = S[i].copy()
        if excluir:
            sims[i] = -1  # evita que a UC compare com ela mesma
        idx = np.argsort(-sims)[:k]
        for rank, j in enumerate(idx, start=1):
            rows.append({
                "UC (Objetivo)": nomes[i],
                "UC (Competência)": nomes[j],
                "Rank": rank,
                "Similaridade": float(S[i, j])
            })

    df_out = pd.DataFrame(rows)

    # -----------------------------------------------------------
    # 📊 Visualizações e métricas
    # -----------------------------------------------------------
    st.markdown("### 📈 Tabela de Alinhamento Top-k")
    st.dataframe(df_out.head(500), use_container_width=True)

    # Cálculo de similaridade média por UC
    mean_sim = df_out.groupby("UC (Objetivo)")["Similaridade"].mean().reset_index()
    mean_sim.rename(columns={"Similaridade": "Similaridade Média"}, inplace=True)
    mean_sim = mean_sim.sort_values(by="Similaridade Média", ascending=False)

    st.markdown("### 🧩 Ranking Geral de Coerência")
    st.dataframe(mean_sim, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(mean_sim["Similaridade Média"], bins=12, color="#3b5bdb", ax=ax, kde=True)
    ax.set_title("Distribuição da Similaridade Média entre Objetivos e Competências")
    ax.set_xlabel("Similaridade Média (0 a 1)")
    ax.set_ylabel("Quantidade de UCs")
    st.pyplot(fig, use_container_width=True)

    # -----------------------------------------------------------
    # 💾 Exportação
    # -----------------------------------------------------------
    export_table(scope_key, df_out, "alinhamento_topk", "Alinhamento Top-k (Objetivos × Competências)")
    export_table(scope_key, mean_sim, "alinhamento_medias", "Similaridade Média por UC")
    export_zip_button(scope_key)

    # -----------------------------------------------------------
    # 📘 Instruções de interpretação
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("📘 Como interpretar os resultados")

    st.markdown(
        """
        **1️⃣ Interpretação dos valores:**
        - **Similaridade ≥ 0.85** → Alinhamento **excelente** (objetivos refletem fortemente as competências).
        - **Entre 0.65 e 0.85** → Alinhamento **adequado**, coerência geral satisfatória.
        - **Entre 0.50 e 0.65** → Alinhamento **moderado**, pode haver dispersão conceitual.
        - **Abaixo de 0.50** → Alinhamento **fraco**, objetivos e competências expressam dimensões distintas.

        **2️⃣ Como analisar no contexto curricular:**
        - UCs com alta coerência reforçam a clareza entre o que se aprende e o que se espera como resultado.
        - UCs com baixa coerência podem indicar **inconsistência de formulação**, necessidade de revisão
          de **verbos de ação**, **níveis de Bloom** ou **congruência de escopo** entre as dimensões formativas.
        - A comparação entre cursos diferentes pode revelar **padrões institucionais de escrita pedagógica**.

        **3️⃣ Uso prático:**
        - Combine esta análise com a *Curva Bloom Progressiva* e *Cobertura Curricular* para identificar
          lacunas ou redundâncias em níveis cognitivos e competências repetidas.
        """
    )
