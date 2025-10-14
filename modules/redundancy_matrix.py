# ===============================================================
# 🧭 EmentaLabv2 — Matriz de Similaridade (Objetos × Competências & DCN)
# ===============================================================
# - Mede o alinhamento semântico entre o que é ensinado (Objetos)
#   e o que se espera do egresso (Competências / DCN)
# - Inclui análise automática via GPT (pontos fortes e fracos)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI
from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, export_zip_button
from utils.text_utils import find_col


def run_alignment_matrix(df, scope_key, client=None):
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

    # -----------------------------------------------------------
    # 📂 Identifica colunas relevantes
    # -----------------------------------------------------------
    col_obj = find_col(df, "Objetos de conhecimento")
    col_comp = find_col(df, "Competências do Perfil do Egresso")
    col_dcn = find_col(df, "Competências DCN")

    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' não encontrada.")
        return

    if not (col_comp or col_dcn):
        st.error("Nenhuma coluna de competências encontrada ('Competências do Egresso' ou 'Competências DCN').")
        return

    # -----------------------------------------------------------
    # 🧹 Pré-processamento
    # -----------------------------------------------------------
    df_valid = df.copy()
    df_valid = df_valid.fillna("")

    textos_obj = df_valid[col_obj].astype(str).tolist()
    nomes = df_valid["Nome da UC"].astype(str).tolist()

    emb_obj = l2_normalize(sbert_embed(textos_obj))

    results = []

    # -----------------------------------------------------------
    # 🧮 Similaridade Objetos × Competências Egresso
    # -----------------------------------------------------------
    if col_comp:
        textos_comp = df_valid[col_comp].astype(str).tolist()
        emb_comp = l2_normalize(sbert_embed(textos_comp))
        sim_comp = np.diag(np.dot(emb_obj, emb_comp.T))
        results.append(("Objetos × Competências Egresso", sim_comp))

    # -----------------------------------------------------------
    # 🧮 Similaridade Objetos × Competências DCN
    # -----------------------------------------------------------
    if col_dcn:
        textos_dcn = df_valid[col_dcn].astype(str).tolist()
        emb_dcn = l2_normalize(sbert_embed(textos_dcn))
        sim_dcn = np.diag(np.dot(emb_obj, emb_dcn.T))
        results.append(("Objetos × Competências DCN", sim_dcn))

    # -----------------------------------------------------------
    # 📊 Monta DataFrame consolidado
    # -----------------------------------------------------------
    if not results:
        st.warning("Nenhuma análise pôde ser realizada — faltam colunas de competências.")
        return

    df_res = pd.DataFrame({"UC": nomes})
    for label, vals in results:
        df_res[label] = vals

    # Remove linhas totalmente vazias
    df_res = df_res.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    st.markdown("### 📈 Similaridade entre Dimensões")
    st.dataframe(df_res, use_container_width=True)
    export_table(scope_key, df_res, "matriz_objetos_competencias", "Matriz Objetos × Competências/DCN")

    # -----------------------------------------------------------
    # 🌡️ Visualização (Heatmap)
    # -----------------------------------------------------------
    st.markdown("### 🌡️ Mapa de Calor de Alinhamento")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_res.set_index("UC"), annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Similaridade (Objetos × Competências / DCN)")
    st.pyplot(fig, use_container_width=True)

    # -----------------------------------------------------------
    # 🧠 Análise interpretativa automática via GPT
    # -----------------------------------------------------------
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
        mean_cols = {col: df_res[col].mean() for col in df_res.columns if col != "UC"}
        resumo = {
            "num_ucs": len(df_res),
            "medias": mean_cols,
            "uc_criticas": df_res[df_res.iloc[:, 1:].mean(axis=1) < 0.65]["UC"].tolist(),
        }

        prompt = f"""
        Você é um avaliador curricular.
        Analise os resultados da matriz de similaridade a seguir (JSON):
        {resumo}

        Gere um relatório técnico curto e objetivo com:
        1️⃣ **Pontos fortes** do alinhamento curricular.
        2️⃣ **Fragilidades** detectadas.
        3️⃣ **Recomendações práticas** para aprimorar coerência entre objetos e competências.

        - Linguagem técnica e concisa.
        - Evite redundância.
        - Máximo de 150 palavras.
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
        st.info("🔑 Chave da OpenAI não encontrada — relatório analítico não foi gerado.")

    # -----------------------------------------------------------
    # 🧭 Interpretação
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("📘 Como interpretar os resultados")
    st.markdown(
        """
        - **≥ 0.85:** Forte coerência entre o que é ensinado e o que se espera do egresso.  
        - **0.65–0.85:** Coerência moderada; há convergência geral, mas com dispersões temáticas.  
        - **< 0.65:** Baixa coerência; revisar objetivos e competências para garantir aderência.  

        💡 **Dica:** UCs com valores baixos simultaneamente em *Objetos × Competências do Egresso* e *Objetos × DCN*
        devem ser priorizadas na revisão curricular e nos planos de aprendizagem.
        """
    )

    export_zip_button(scope_key)
