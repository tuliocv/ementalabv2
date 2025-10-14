# ===============================================================
# 🧬 EmentaLabv2 — Redundância e Análise Frase-a-Frase (v9.3)
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
    st.header("🧬 Redundância entre UCs")
    st.caption(
        """
        Esta análise identifica **similaridades excessivas de conteúdo entre as Unidades Curriculares (UCs)**.
        Utiliza embeddings semânticos (SBERT) para comparar ementas e detectar redundâncias de temas,
        conceitos ou objetivos de aprendizagem.  
        Valores de similaridade altos indicam UCs potencialmente **sobrepostas ou repetitivas**.
        """
    )

    col_base = find_col(df, "Ementa") or find_col(df, "Objetos de conhecimento")
    if not col_base:
        st.error("Coluna de texto principal ('Ementa' ou 'Objetos de conhecimento') não encontrada.")
        st.stop()

    textos = df[col_base].astype(str).apply(replace_semicolons).tolist()
    nomes = df["Nome da UC"].astype(str).tolist()

    with st.spinner("🧠 Calculando embeddings e matriz de similaridade SBERT..."):
        emb = l2_normalize(sbert_embed(textos))
        S = np.dot(emb, emb.T)

    st.markdown("### 🧮 Matriz de Similaridade Global")
    df_mat = pd.DataFrame(S, index=nomes, columns=nomes)
    st.dataframe(
        df_mat.head(30)
        .style.format("{:.2f}")
        .background_gradient(cmap="RdYlGn_r", vmin=0, vmax=1),
        use_container_width=True,
    )
    export_table(scope_key, df_mat, "redundancia_matriz", "Matriz de Similaridade entre UCs")

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
        - Utilize esta matriz para revisar o **portfólio de UCs** e evitar sobreposições temáticas.  
        - Avalie se redundâncias representam **reforço formativo (positivo)** ou **repetição desnecessária (negativo)**.  
        """
    )


# ---------------------------------------------------------------
# 🧩 Comparação Frase-a-Frase entre duas UCs
# ---------------------------------------------------------------
def run_pair_analysis(df, scope_key):
    st.header("🔬 Análise Frase a Frase entre UCs")
    st.caption(
        """
        Permite uma **comparação semântica linha a linha** entre duas ementas selecionadas,
        destacando trechos semelhantes.  
        Útil para investigar redundâncias detectadas na análise global e entender
        **quais passagens se repetem entre UCs**.
        """
    )

    col_base = find_col(df, "Ementa") or find_col(df, "Objetos de conhecimento")
    if not col_base:
        st.error("Coluna de texto não encontrada.")
        st.stop()

    nomes = df["Nome da UC"].dropna().unique().tolist()
    uc_a = st.selectbox("📘 UC A", nomes)
    uc_b = st.selectbox("📗 UC B", [n for n in nomes if n != uc_a])

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

    st.markdown("### 🧩 Trechos mais semelhantes")
    st.dataframe(
        df_out.head(15).style.format({"Similaridade": "{:.3f}"}),
        use_container_width=True,
    )
    export_table(scope_key, df_out, f"redundancia_{uc_a}_vs_{uc_b}", "Redundância Frase a Frase")
    export_zip_button(scope_key)

    st.markdown("---")
    st.subheader("📘 Como interpretar")
    st.markdown(
        f"""
        **1️⃣ O que observar entre '{uc_a}' e '{uc_b}':**
        - Similaridade ≥ 0.85 → repetição literal.  
        - 0.65–0.85 → paráfrases conceituais.  
        - Abaixo de 0.65 → apenas relação tangencial.  

        **2️⃣ Aplicações práticas:**
        - Localizar duplicações e **revisar UCs redundantes**.  
        - Apoiar decisões de **fusão ou reformulação textual**.  
        """
    )


# ---------------------------------------------------------------
# 🧭 Matriz de Similaridade — Objetos × Competências & DCN
# ---------------------------------------------------------------
def run_alignment_matrix(df, scope_key):
    st.header("🧭 Matriz de Similaridade (Objetos × Competências & DCN)")
    st.caption(
        """
        Mede o quanto cada UC está semanticamente **alinhada** entre:
        - **Objetos de Conhecimento × Competências do Egresso**  
        - **Objetos de Conhecimento × Competências das DCNs**

        Valores próximos de **1.00** indicam **forte coerência** entre o que é ensinado,
        o perfil do egresso e as competências normativas das DCNs.
        """
    )

    col_obj = find_col(df, "Objetos de conhecimento")
    col_comp = find_col(df, "Competências do Perfil do Egresso")
    col_dcn = find_col(df, "Competências DCN")

    if not col_obj or not (col_comp or col_dcn):
        st.error("É necessário ter colunas de 'Objetos de conhecimento' e 'Competências' (Egresso/DCN).")
        return

    df_valid = df.dropna(subset=[col_obj])
    textos_obj = df_valid[col_obj].astype(str).tolist()
    textos_comp = df_valid[col_comp].astype(str).tolist() if col_comp else None
    textos_dcn = df_valid[col_dcn].astype(str).tolist() if col_dcn else None
    nomes = df_valid["Nome da UC"].astype(str).tolist()

    emb_obj = l2_normalize(sbert_embed(textos_obj))
    results = []

    if textos_comp:
        emb_comp = l2_normalize(sbert_embed(textos_comp))
        sim_comp = np.diag(np.dot(emb_obj, emb_comp.T))
        results.append(("Objetos × Competências Egresso", sim_comp))
    if textos_dcn:
        emb_dcn = l2_normalize(sbert_embed(textos_dcn))
        sim_dcn = np.diag(np.dot(emb_obj, emb_dcn.T))
        results.append(("Objetos × Competências DCN", sim_dcn))

    df_res = pd.DataFrame({"UC": nomes})
    for label, vals in results:
        df_res[label] = vals

    st.markdown("### 📈 Similaridade entre Dimensões")
    st.dataframe(df_res, use_container_width=True)
    export_table(scope_key, df_res, "matriz_objetos_competencias", "Matriz Objetos × Competências/DCN")

    st.markdown("### 🌡️ Mapa de Calor de Alinhamento")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_res.set_index("UC"), annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Similaridade (Objetos × Competências / DCN)")
    st.pyplot(fig, use_container_width=True)

    export_zip_button(scope_key)

    st.markdown("---")
    st.subheader("📘 Como interpretar os resultados")
    st.markdown(
        """
        - **≥ 0.85:** Forte coerência entre o que é ensinado e o que se espera formar.  
        - **0.65–0.85:** Coerência moderada; há alinhamento geral, mas com dispersões.  
        - **< 0.65:** Baixa coerência; conteúdos e competências podem estar desconectados.  

        💡 **Dica:** UCs com baixa correlação simultânea entre *Objetos × Competências do Egresso* e *Objetos × DCN* devem ser revisadas quanto à clareza de objetivos e aderência normativa.
        """
    )
