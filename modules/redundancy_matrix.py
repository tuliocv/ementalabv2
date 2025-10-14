# ===============================================================
# 🧬 EmentaLabv2 — Redundância e Análise Frase-a-Frase (v9.2)
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, export_zip_button
from utils.text_utils import (
    find_col,
    replace_semicolons,
    _split_sentences
)

# ---------------------------------------------------------------
# 🔁 Análise de Redundância Global
# ---------------------------------------------------------------
def run_redundancy(df, scope_key):
    # -----------------------------------------------------------
    # 🏷️ Título e descrição contextual
    # -----------------------------------------------------------
    st.header("🧬 Redundância entre UCs")
    st.caption(
        """
        Esta análise identifica **similaridades excessivas de conteúdo entre as Unidades Curriculares (UCs)**.
        Utiliza embeddings semânticos (SBERT) para comparar ementas e detectar redundâncias de temas,
        conceitos ou objetivos de aprendizagem.  
        Valores de similaridade altos indicam UCs potencialmente **sobrepostas ou repetitivas**.
        """
    )

    # -----------------------------------------------------------
    # 📂 Coluna base
    # -----------------------------------------------------------
    col_base = find_col(df, "Ementa") or find_col(df, "Objetos de conhecimento")
    if not col_base:
        st.error("Coluna de texto principal ('Ementa' ou 'Objetos de conhecimento') não encontrada.")
        st.stop()

    textos = df[col_base].astype(str).apply(replace_semicolons).tolist()
    nomes = df["Nome da UC"].astype(str).tolist()

    # -----------------------------------------------------------
    # 🔢 Cálculo da matriz de similaridade
    # -----------------------------------------------------------
    with st.spinner("🧠 Calculando embeddings e matriz de similaridade SBERT..."):
        emb = l2_normalize(sbert_embed(textos))
        S = np.dot(emb, emb.T)

    # -----------------------------------------------------------
    # 📈 Visualização da matriz
    # -----------------------------------------------------------
    st.markdown("### 🧮 Matriz de Similaridade Global")
    df_mat = pd.DataFrame(S, index=nomes, columns=nomes)
    st.dataframe(
        df_mat.head(30)
        .style.format("{:.2f}")
        .background_gradient(cmap="RdYlGn_r", vmin=0, vmax=1),
        use_container_width=True,
    )
    export_table(scope_key, df_mat, "redundancia_matriz", "Matriz de Similaridade entre UCs")

    # -----------------------------------------------------------
    # 📊 Identificação de pares redundantes
    # -----------------------------------------------------------
    st.markdown("### 🔗 Pares de UCs com alta similaridade")
    thr = st.slider("Limiar de redundância", 0.5, 0.95, 0.8)
    pares = []
    n = S.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= thr:
                pares.append(
                    {"UC A": nomes[i], "UC B": nomes[j], "Similaridade": float(S[i, j])}
                )
    df_pares = pd.DataFrame(pares)
    if not df_pares.empty:
        df_pares = df_pares.sort_values("Similaridade", ascending=False)
        st.dataframe(df_pares.head(100), use_container_width=True)

        # Distribuição das similaridades
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df_pares["Similaridade"], bins=10, kde=True, color="#3b5bdb", ax=ax)
        ax.set_title("Distribuição das Similaridades (UCs Redundantes)")
        ax.set_xlabel("Similaridade")
        ax.set_ylabel("Frequência")
        st.pyplot(fig, use_container_width=True)

        export_table(scope_key, df_pares, "redundancia_pares", "Pares de UCs Redundantes")
    else:
        st.info("Nenhum par de UCs com similaridade acima do limiar escolhido.")

    export_zip_button(scope_key)

    # -----------------------------------------------------------
    # 📘 Interpretação dos resultados
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("📘 Como interpretar os resultados")

    st.markdown(
        """
        **1️⃣ Interpretação dos valores:**
        - **Similaridade ≥ 0.90** → Altíssima redundância (UCs possivelmente duplicadas).
        - **Entre 0.75 e 0.90** → Redundância alta (conteúdos muito próximos, possível sobreposição).
        - **Entre 0.60 e 0.75** → Similaridade média (áreas afins ou interdisciplinaridade natural).
        - **Abaixo de 0.60** → Similaridade baixa (UCs bem diferenciadas).

        **2️⃣ Como analisar:**
        - Utilize esta matriz para **revisar o portfólio de UCs**, evitando sobreposições temáticas
          entre diferentes períodos, cursos ou núcleos.
        - Em clusters de alta redundância, avalie a possibilidade de **integração de conteúdos**.
        - Combine esta análise com *Convergência Temática* e *Cobertura Curricular* para compreender
          se a redundância é **intencional (reforço formativo)** ou **problemática (repetição desnecessária)**.
        """
    )

# ---------------------------------------------------------------
# 🧩 Comparação Frase-a-Frase entre duas UCs
# ---------------------------------------------------------------
def run_pair_analysis(df, scope_key):
    """Comparação detalhada frase a frase entre duas UCs"""
    st.header("🔬 Análise Frase a Frase entre UCs")
    st.caption(
        """
        Permite uma **comparação semântica linha a linha** entre duas ementas selecionadas,
        destacando trechos textualmente semelhantes.  
        É útil para investigar redundâncias detectadas na análise global e compreender
        **quais passagens específicas se repetem entre as UCs**.
        """
    )

    # -----------------------------------------------------------
    # 📂 Seleção de UCs
    # -----------------------------------------------------------
    col_base = find_col(df, "Ementa") or find_col(df, "Objetos de conhecimento")
    if not col_base:
        st.error("Coluna de texto não encontrada.")
        st.stop()

    nomes = df["Nome da UC"].dropna().unique().tolist()
    uc_a = st.selectbox("📘 UC A", nomes)
    uc_b = st.selectbox("📗 UC B", [n for n in nomes if n != uc_a])

    # -----------------------------------------------------------
    # 🧠 Similaridade entre frases
    # -----------------------------------------------------------
    text_a = replace_semicolons(df.loc[df["Nome da UC"] == uc_a, col_base].iloc[0])
    text_b = replace_semicolons(df.loc[df["Nome da UC"] == uc_b, col_base].iloc[0])
    ph_a, ph_b = _split_sentences(text_a), _split_sentences(text_b)

    if not ph_a or not ph_b:
        st.warning("Não há frases suficientes para comparar.")
        return

    emb_a, emb_b = sbert_embed(ph_a), sbert_embed(ph_b)
    sim = cosine_similarity(emb_a, emb_b)

    rows = []
    for i in range(len(ph_a)):
        j = np.argmax(sim[i])
        rows.append(
            {"Similaridade": sim[i, j], "Trecho A": ph_a[i], "Trecho B": ph_b[j]}
        )
    df_out = pd.DataFrame(rows).sort_values("Similaridade", ascending=False)

    # -----------------------------------------------------------
    # 📈 Exibição dos resultados
    # -----------------------------------------------------------
    st.markdown("### 🧩 Trechos mais semelhantes")
    st.dataframe(
        df_out.head(15).style.format({"Similaridade": "{:.3f}"}),
        use_container_width=True,
    )

    export_table(scope_key, df_out, f"redundancia_{uc_a}_vs_{uc_b}", "Redundância Frase a Frase")
    export_zip_button(scope_key)

    # -----------------------------------------------------------
    # 📘 Interpretação e uso prático
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("📘 Como interpretar")

    st.markdown(
        f"""
        **1️⃣ O que observar entre '{uc_a}' e '{uc_b}':**
        - Trechos com similaridade **≥ 0.85** indicam repetição quase literal.
        - Similaridades entre **0.65 e 0.85** podem representar **paráfrases conceituais** (mesmo conteúdo reescrito).
        - Abaixo de **0.65** o conteúdo tende a ser apenas tangencialmente relacionado.

        **2️⃣ Aplicações práticas:**
        - Use esta análise para **localizar duplicações** de conceitos, definições ou objetivos.
        - Apoia decisões de **reorganização curricular**, fusão de UCs ou reformulação textual.
        - Pode servir como **evidência em processos de revisão de PPC** e otimização de carga horária.
        """
    )
