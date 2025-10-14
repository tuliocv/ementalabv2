# ===============================================================
# 🧬 EmentaLabv2 — Similaridade, Redundância e Alinhamento (v12.0)
# ===============================================================
# Inclui:
# - 🔁 run_redundancy: redundância entre UCs (global)
# - 🔬 run_pair_analysis: comparação frase a frase
# - 🧭 run_alignment_matrix: alinhamento Objetos × Competências & DCN (robusta)
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
    st.dataframe(
        df_mat.head(30)
        .style.format("{:.2f}")
        .background_gradient(cmap="RdYlGn_r", vmin=0, vmax=1),
        use_container_width=True,
    )
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
        st.dataframe(df_pares.head(100), use_container_width=True)
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
            "Similaridade": s_row[j],
            "Trecho A": ph_a[i],
            "Trecho B": ph_b[j]
        })
    df_out = pd.DataFrame(rows).sort_values("Similaridade", ascending=False)

    st.markdown("### 🧩 Trechos mais semelhantes")
    st.dataframe(df_out.head(15).style.format({"Similaridade": "{:.3f}"}), use_container_width=True)
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
    st.header("🧭 Matriz de Similaridade — Objetos × Competências & DCN")
    st.caption(
        """
        Mede o quanto cada UC está semanticamente **alinhada** entre:
        - **Objetos de Conhecimento × Competências do Egresso**  
        - **Objetos de Conhecimento × Relação DCN**

        Valores próximos de **1.00** indicam **forte coerência** entre o que é ensinado,
        o perfil do egresso e as competências normativas das DCNs.
        """
    )

    col_obj = find_col(df, "Objetos de conhecimento")
    col_comp = find_col(df, "Competências do Perfil do Egresso")
    col_rel_dcn = find_col(df, "Relação DCN") or find_col(df, "Competências DCN")

    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' não encontrada.")
        return
    if not (col_comp or col_rel_dcn):
        st.error("Nenhuma coluna de competências encontrada ('Egresso' ou 'Relação DCN').")
        return

    df_valid = df.fillna("")
    nomes = df_valid["Nome da UC"].astype(str).tolist()

    # ---------- Cálculo robusto de similaridade ----------
    def calculate_aggregate_similarity(df_scope, col_objetos, col_comp, col_rel_dcn):
        campos_chave = {
            "Competências do Perfil do Egresso": col_comp,
            "Objetos de conhecimento": col_objetos,
            "Relação DCN": col_rel_dcn,
        }
        cols_validas = [c for c in campos_chave.values() if c and c in df_scope.columns]
        if len(cols_validas) < 2:
            return None, "Dados insuficientes."

        textos = {
            k: df_scope[v].fillna("").astype(str).apply(replace_semicolons).tolist()
            for k, v in campos_chave.items() if v in df_scope.columns
        }
        emb = {k: l2_normalize(sbert_embed(v)) for k, v in textos.items()}

        metrics = {}
        if 'Objetos de conhecimento' in emb and 'Competências do Perfil do Egresso' in emb:
            sim = np.sum(emb['Objetos de conhecimento'] * emb['Competências do Perfil do Egresso'], axis=1)
            metrics['Sim. Obj × Comp Egresso'] = sim
        if 'Objetos de conhecimento' in emb and 'Relação DCN' in emb:
            sim = np.sum(emb['Objetos de conhecimento'] * emb['Relação DCN'], axis=1)
            metrics['Sim. Obj × DCN'] = sim

        if not metrics:
            return None, "Não foi possível calcular similaridade."
        return metrics, None

    with st.spinner("🧠 Calculando similaridades semânticas..."):
        metrics, err = calculate_aggregate_similarity(df_valid, col_obj, col_comp, col_rel_dcn)
        if err:
            st.error(err)
            return

    df_res = pd.DataFrame({"UC": nomes})
    for k, v in metrics.items():
        df_res[k] = v
    df_res = df_res.replace([np.inf, -np.inf], np.nan).fillna(0)

    st.markdown("### 📈 Similaridade entre Dimensões")
    st.dataframe(df_res.style.format("{:.2f}"), use_container_width=True)
    export_table(scope_key, df_res, "matriz_objetos_competencias", "Matriz Objetos × Competências/DCN")

    # ---------- Gráfico ----------
    st.markdown("### 🌡️ Mapa de Calor de Alinhamento")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_res.set_index("UC"), annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Similaridade (Objetos × Competências / DCN)")
    st.pyplot(fig, use_container_width=True)

    # ---------- Relatório GPT ----------
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
            "médias": {col: float(df_res[col].mean()) for col in df_res.columns if col != "UC"},
            "ucs_baixas": df_res[df_res.iloc[:, 1:].mean(axis=1) < 0.65]["UC"].tolist(),
            "total_ucs": len(df_res),
        }
        prompt = f"""
        Você é um avaliador curricular.
        Analise os resultados da matriz de similaridade abaixo:
        {resumo}

        Gere um relatório técnico curto e direto, destacando:
        - Pontos fortes
        - Fragilidades
        - Recomendações práticas

        Linguagem técnica, objetiva e sem redundância.
        """
        try:
            with st.spinner("📄 Gerando relatório via GPT..."):
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
        st.info("🔑 Chave da OpenAI não configurada — relatório não gerado.")

    # ---------- Interpretação ----------
    st.markdown("---")
    st.subheader("📘 Como interpretar os resultados")
    st.markdown(
        """
        - **≥ 0.85:** Forte coerência entre o que é ensinado e as competências.  
        - **0.65–0.85:** Coerência moderada; há convergência geral, mas com dispersões.  
        - **< 0.65:** Baixa coerência; revisar descrição dos objetos e competências.  

        💡 **Dica:** UCs com baixa coerência simultânea em *Objetos × Competências do Egresso* e *Objetos × DCN*
        devem ser priorizadas em revisões curriculares.
        """
    )

    export_zip_button(scope_key)
