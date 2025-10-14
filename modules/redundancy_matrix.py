# ===============================================================
# 🧬 EmentaLabv2 — Similaridade, Redundância e Alinhamento (v11.3)
# ===============================================================
# Inclui:
# - 🔁 run_redundancy: redundância entre UCs (global)
# - 🔬 run_pair_analysis: comparação frase a frase
# - 🧭 run_alignment_matrix: alinhamento Objetos × Competências & DCN
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, export_zip_button
from utils.text_utils import find_col, replace_semicolons, _split_sentences


# ===============================================================
# 🧩 Função auxiliar para formatação segura
# ===============================================================
def safe_style(df, decimals=2, cmap="YlGnBu"):
    """
    Garante que apenas colunas numéricas sejam formatadas.
    Evita erro 'Unknown format code f for object of type str'.
    """
    df_fmt = df.copy()
    # converte todas as colunas (exceto a primeira, geralmente UC)
    for c in df_fmt.columns:
        if c != df_fmt.columns[0]:
            df_fmt[c] = pd.to_numeric(df_fmt[c], errors="coerce")
    # aplica formatação apenas às numéricas
    fmt_dict = {
        c: f"{{:.{decimals}f}}" for c in df_fmt.columns if pd.api.types.is_numeric_dtype(df_fmt[c])
    }
    return df_fmt.style.format(fmt_dict).background_gradient(cmap=cmap, vmin=0, vmax=1)


# ===============================================================
# 🔁 1. Análise de Redundância Global
# ===============================================================
def run_redundancy(df, scope_key):
    """Detecta similaridades excessivas entre UCs."""
    st.header("🧬 Redundância entre UCs")
    st.caption(
        """
        Esta análise identifica **similaridades excessivas** de conteúdo entre Unidades Curriculares (UCs),
        comparando suas ementas ou objetos de conhecimento via embeddings SBERT.
        Valores altos indicam UCs potencialmente **repetitivas ou sobrepostas**.
        """
    )

    col_base = find_col(df, "Ementa") or find_col(df, "Objetos de conhecimento")
    if not col_base:
        st.error("Coluna 'Ementa' ou 'Objetos de conhecimento' não encontrada.")
        return

    textos = df[col_base].astype(str).apply(replace_semicolons).tolist()
    nomes = df["Nome da UC"].astype(str).tolist()

    with st.spinner("🧠 Calculando embeddings e similaridades..."):
        emb = l2_normalize(sbert_embed(textos))
        S = np.dot(emb, emb.T)

    df_mat = pd.DataFrame(S, index=nomes, columns=nomes)
    st.markdown("### 🧮 Matriz de Similaridade Global")

    st.dataframe(safe_style(df_mat.head(30), 2, "RdYlGn_r"), use_container_width=True)
    export_table(scope_key, df_mat, "redundancia_matriz", "Matriz de Similaridade entre UCs")

    thr = st.slider("Limiar de redundância (similaridade mínima)", 0.5, 0.95, 0.8, 0.05)
    pares = [
        {"UC A": nomes[i], "UC B": nomes[j], "Similaridade": float(S[i, j])}
        for i in range(len(S)) for j in range(i + 1, len(S))
        if S[i, j] >= thr
    ]

    if pares:
        df_pares = pd.DataFrame(pares).sort_values("Similaridade", ascending=False)
        st.markdown("### 🔗 Pares de UCs com alta similaridade")
        st.dataframe(safe_style(df_pares, 3, "YlOrRd"), use_container_width=True)
        export_table(scope_key, df_pares, "redundancia_pares", "UCs com alta similaridade")

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df_pares["Similaridade"], bins=10, kde=True, color="#3b5bdb", ax=ax)
        ax.set_title("Distribuição das Similaridades (UCs Redundantes)")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Nenhum par de UCs com similaridade acima do limiar escolhido.")

    export_zip_button(scope_key)

    st.markdown("---")
    st.subheader("📘 Interpretação")
    st.markdown(
        """
        - **≥ 0.90:** UCs possivelmente duplicadas.  
        - **0.75–0.90:** Alta redundância → revisar sobreposição.  
        - **0.60–0.75:** Similaridade moderada (interdisciplinaridade natural).  
        - **< 0.60:** Diferenciação adequada.  
        """
    )


# ===============================================================
# 🔬 2. Comparação Frase a Frase
# ===============================================================
def run_pair_analysis(df, scope_key):
    """Comparação semântica linha a linha entre duas UCs."""
    st.header("🔬 Análise Frase a Frase entre UCs")
    st.caption(
        """
        Compara duas UCs frase a frase, destacando **trechos textualmente semelhantes**.
        Útil para verificar redundâncias específicas e reformular conteúdos sobrepostos.
        """
    )

    col_base = find_col(df, "Ementa") or find_col(df, "Objetos de conhecimento")
    if not col_base:
        st.error("Coluna de texto não encontrada.")
        return

    nomes = df["Nome da UC"].dropna().unique().tolist()
    uc_a = st.selectbox("📘 UC A", nomes)
    uc_b = st.selectbox("📗 UC B", [n for n in nomes if n != uc_a])

    text_a = replace_semicolons(df.loc[df["Nome da UC"] == uc_a, col_base].iloc[0])
    text_b = replace_semicolons(df.loc[df["Nome da UC"] == uc_b, col_base].iloc[0])

    ph_a, ph_b = _split_sentences(text_a), _split_sentences(text_b)
    if not ph_a or not ph_b:
        st.warning("Textos insuficientes para análise.")
        return

    emb_a, emb_b = sbert_embed(ph_a), sbert_embed(ph_b)
    sim = cosine_similarity(emb_a, emb_b)

    rows = []
    for i, s_row in enumerate(sim):
        j = np.argmax(s_row)
        rows.append({
            "Similaridade": float(s_row[j]),
            "Trecho A": ph_a[i],
            "Trecho B": ph_b[j]
        })
    df_out = pd.DataFrame(rows).sort_values("Similaridade", ascending=False)

    st.markdown("### 🧩 Trechos mais semelhantes")
    st.dataframe(safe_style(df_out, 3, "PuBu"), use_container_width=True)
    export_table(scope_key, df_out, f"redundancia_{uc_a}_vs_{uc_b}", f"Redundância Frase a Frase: {uc_a} vs {uc_b}")
    export_zip_button(scope_key)

    st.markdown("---")
    st.subheader("📘 Interpretação")
    st.markdown(
        f"""
        - **≥ 0.85:** Repetição literal.  
        - **0.65–0.85:** Parafraseamento conceitual.  
        - **< 0.65:** Relação tangencial.  
        """
    )


# ===============================================================
# 🧭 3. Matriz de Similaridade (Objetos × Competências & DCN)
# ===============================================================
def run_alignment_matrix(df, scope_key, client=None):
    """Avalia alinhamento semântico entre Objetos, Competências e DCN."""
    st.header("🧭 Matriz de Similaridade — Objetos × Competências & DCN")
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

    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' não encontrada.")
        return
    if not (col_comp or col_dcn):
        st.error("Nenhuma coluna de competências encontrada.")
        return

    df_valid = df.fillna("")
    textos_obj = df_valid[col_obj].astype(str).tolist()
    nomes = df_valid["Nome da UC"].astype(str).tolist()
    emb_obj = l2_normalize(sbert_embed(textos_obj))

    results = []
    if col_comp:
        emb_comp = l2_normalize(sbert_embed(df_valid[col_comp].astype(str).tolist()))
        results.append(("Objetos × Competências Egresso", np.diag(np.dot(emb_obj, emb_comp.T))))
    if col_dcn:
        emb_dcn = l2_normalize(sbert_embed(df_valid[col_dcn].astype(str).tolist()))
        results.append(("Objetos × Competências DCN", np.diag(np.dot(emb_obj, emb_dcn.T))))

    df_res = pd.DataFrame({"UC": nomes})
    for label, vals in results:
        df_res[label] = vals

    st.markdown("### 📈 Similaridade entre Dimensões")
    st.dataframe(safe_style(df_res, 2, "YlGnBu"), use_container_width=True)
    export_table(scope_key, df_res, "matriz_objetos_competencias", "Matriz Objetos × Competências/DCN")

    # 🔹 Heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_res.set_index("UC"), annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Similaridade (Objetos × Competências / DCN)")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, use_container_width=True)

    # -------------------- GPT Relatório --------------------
    if client is None:
        api_key = st.session_state.get("global_api_key", "")
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
            except Exception:
                client = None

    st.markdown("---")
    st.subheader("🧾 Relatório Analítico de Alinhamento Curricular")

    if client:
        resumo = {
            "media_egresso": float(df_res["Objetos × Competências Egresso"].mean()) if "Objetos × Competências Egresso" in df_res else None,
            "media_dcn": float(df_res["Objetos × Competências DCN"].mean()) if "Objetos × Competências DCN" in df_res else None,
            "ucs_baixas": df_res[df_res.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').mean(axis=1) < 0.65]["UC"].tolist()
        }
        prompt = f"""
        Você é um avaliador curricular. Analise os seguintes dados:
        {resumo}

        Gere um relatório técnico curto (máx. 150 palavras) com:
        - Pontos fortes
        - Fragilidades
        - Recomendações práticas
        Linguagem objetiva e técnica.
        """
        try:
            with st.spinner("🧠 Gerando relatório via GPT..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            analise = resp.choices[0].message.content.strip()
            st.success("Relatório gerado com sucesso.")
            st.markdown(analise)
        except Exception as e:
            st.error(f"Erro ao gerar relatório via GPT: {e}")
    else:
        st.info("🔑 Chave da OpenAI não encontrada — relatório não gerado.")

    export_zip_button(scope_key)
