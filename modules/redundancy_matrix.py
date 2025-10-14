# ===============================================================
# 🧬 EmentaLabv2 — Similaridade, Redundância e Alinhamento (v11.4)
# ===============================================================
# Novo recurso:
# - 🧭 Matriz única de similaridade (Objetos × Egresso × DCN)
# - Alternar entre modo “médio por UC” e “frase a frase”
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
# 🔧 Função auxiliar de formatação
# ===============================================================
def safe_style(df, cmap="RdYlGn", decimals=2):
    df_fmt = df.copy()
    for c in df_fmt.columns[1:]:
        df_fmt[c] = pd.to_numeric(df_fmt[c], errors="coerce")
    fmt_dict = {c: f"{{:.{decimals}f}}" for c in df_fmt.columns[1:]}
    return df_fmt.style.format(fmt_dict).background_gradient(cmap=cmap, vmin=0, vmax=1)


# ===============================================================
# 🧭 Matriz de Similaridade Integrada (Objetos × Competências & DCN)
# ===============================================================
def run_alignment_matrix(df, scope_key, client=None):
    st.header("🧭 Matriz de Similaridade Integrada — Objetos × Competências & DCN")
    st.caption(
        """
        Mede o quanto cada UC está semanticamente **alinhada** entre:
        - **Objetos de Conhecimento × Competências do Egresso**
        - **Objetos de Conhecimento × Competências das DCNs**

        Valores mais próximos de **1.00 (verde)** indicam **forte coerência** entre
        o que é ensinado, o perfil do egresso e as competências normativas das DCNs.
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
    nomes = df_valid["Nome da UC"].astype(str).tolist()

    # Escolha do modo de análise
    st.markdown("### ⚙️ Configuração da análise")
    modo = st.radio(
        "Selecione o tipo de cálculo de similaridade:",
        ["📊 Por Média da UC", "🧩 Frase a Frase"],
        horizontal=True,
    )

    # -----------------------------------------------------------
    # 🧠 MODO 1: Média da UC (vetor do texto inteiro)
    # -----------------------------------------------------------
    if modo == "📊 Por Média da UC":
        textos_obj = df_valid[col_obj].astype(str).apply(replace_semicolons).tolist()
        emb_obj = l2_normalize(sbert_embed(textos_obj))

        results = {"UC": nomes}
        if col_comp:
            emb_comp = l2_normalize(sbert_embed(df_valid[col_comp].astype(str).tolist()))
            results["Similaridade (Objetos × Egresso)"] = np.diag(np.dot(emb_obj, emb_comp.T))
        if col_dcn:
            emb_dcn = l2_normalize(sbert_embed(df_valid[col_dcn].astype(str).tolist()))
            results["Similaridade (Objetos × DCN)"] = np.diag(np.dot(emb_obj, emb_dcn.T))

        df_res = pd.DataFrame(results)

    # -----------------------------------------------------------
    # 🧩 MODO 2: Frase a Frase (média de similaridades par-a-par)
    # -----------------------------------------------------------
    else:
        rows = []
        for _, row in df_valid.iterrows():
            nome = str(row["Nome da UC"])
            obj_text = replace_semicolons(str(row.get(col_obj, "")))
            comp_text = replace_semicolons(str(row.get(col_comp, "")))
            dcn_text = replace_semicolons(str(row.get(col_dcn, "")))

            # Divisão em frases
            objs = _split_sentences(obj_text)
            comps = _split_sentences(comp_text)
            dcns = _split_sentences(dcn_text)

            # Calcula similaridade média
            emb_obj = sbert_embed(objs)
            sim_egresso = None
            sim_dcn = None

            if comp_text:
                emb_comp = sbert_embed(comps)
                sim_matrix = cosine_similarity(emb_obj, emb_comp)
                sim_egresso = float(np.mean(sim_matrix)) if sim_matrix.size > 0 else np.nan

            if dcn_text:
                emb_dcn = sbert_embed(dcns)
                sim_matrix = cosine_similarity(emb_obj, emb_dcn)
                sim_dcn = float(np.mean(sim_matrix)) if sim_matrix.size > 0 else np.nan

            rows.append({
                "UC": nome,
                "Similaridade (Objetos × Egresso)": sim_egresso,
                "Similaridade (Objetos × DCN)": sim_dcn,
            })
        df_res = pd.DataFrame(rows)

    # -----------------------------------------------------------
    # 📈 Exibição e Heatmap
    # -----------------------------------------------------------
    st.markdown("### 📊 Matriz de Similaridade Combinada")
    st.dataframe(safe_style(df_res, cmap="RdYlGn"), use_container_width=True)
    export_table(scope_key, df_res, "matriz_alinhamento_unificada", "Matriz Integrada de Similaridade")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_res.set_index("UC"), annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Mapa de Similaridade (Objetos × Egresso × DCN)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, use_container_width=True)

    # -----------------------------------------------------------
    # 🧾 Relatório Analítico via GPT
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
        medias = df_res.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").mean()
        ucs_baixas = df_res[df_res.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").mean(axis=1) < 0.65]["UC"].tolist()

        resumo = {
            "média_egresso": float(medias.iloc[0]) if len(medias) > 0 else None,
            "média_dcn": float(medias.iloc[1]) if len(medias) > 1 else None,
            "ucs_baixas": ucs_baixas,
            "modo": modo,
        }

        prompt = f"""
        Você é um avaliador curricular. Analise os seguintes dados:
        {resumo}

        Gere um relatório técnico e direto (até 150 palavras) com:
        - Pontos fortes observados
        - Fragilidades detectadas
        - Recomendações práticas de melhoria

        Mantenha linguagem objetiva e avaliativa, com foco pedagógico.
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

    # -----------------------------------------------------------
    # 🧭 Interpretação final
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("📘 Como interpretar esta matriz")
    st.markdown(
        """
        - **Verde (≥ 0.85):** Forte coerência semântica → objetivos e competências estão bem alinhados.  
        - **Amarelo (0.65–0.85):** Coerência moderada → verificar clareza ou foco dos objetivos.  
        - **Vermelho (< 0.65):** Alinhamento fraco → revisar formulações de objetivos ou vínculo com as DCNs.  

        **Modo “Frase a Frase”** permite identificar desalinhamentos textuais internos,
        enquanto **“Por Média da UC”** dá uma visão mais geral de coerência pedagógica.
        """
    )
