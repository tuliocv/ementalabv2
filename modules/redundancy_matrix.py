# ===============================================================
# 🧬 EmentaLabv2 — Similaridade, Redundância e Alinhamento (v11.7)
# ===============================================================

import re
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, export_zip_button, show_and_export_fig
from utils.text_utils import find_col, replace_semicolons, _split_sentences


# ===============================================================
# 🔧 Estilo seguro para DataFrames numéricos
# ===============================================================
def safe_style(df: pd.DataFrame, cmap="RdYlGn", decimals=2):
    if df.empty:
        return df
    df_fmt = df.copy()
    num_cols = [c for c in df_fmt.columns if c != "UC"]
    for c in num_cols:
        df_fmt[c] = pd.to_numeric(df_fmt[c], errors="coerce")
    fmt_dict = {c: f"{{:.{decimals}f}}" for c in num_cols}
    return df_fmt.style.format(fmt_dict).background_gradient(cmap=cmap, vmin=0, vmax=1)


# ===============================================================
# 🔁 1) Redundância Global entre UCs
# ===============================================================
def run_redundancy(df: pd.DataFrame, scope_key: str):
    st.header("🧬 Redundância entre UCs")
    st.caption(
        "Compara ementas/objetos de conhecimento via embeddings SBERT para detectar "
        "**sobreposição de conteúdo** entre Unidades Curriculares."
    )

    col_base = find_col(df, "Ementa") or find_col(df, "Objetos de conhecimento")
    if not col_base:
        st.error("Coluna 'Ementa' ou 'Objetos de conhecimento' não encontrada.")
        return

    df_an = df.dropna(subset=[col_base]).copy()
    if df_an.empty:
        st.info("Nenhuma UC com texto disponível para análise.")
        return

    nomes = df_an["Nome da UC"].astype(str).tolist()
    textos = df_an[col_base].astype(str).apply(replace_semicolons).tolist()

    with st.spinner("🧠 Calculando embeddings e similaridades..."):
        emb = l2_normalize(sbert_embed(textos))
        S = np.dot(emb, emb.T)

    df_mat = pd.DataFrame(S, index=nomes, columns=nomes)
    st.markdown("### 🧮 Matriz de Similaridade Global (amostra)")
    st.dataframe(
        df_mat.head(30).style.format("{:.2f}").background_gradient(cmap="RdYlGn_r", vmin=0, vmax=1),
        use_container_width=True,
    )
    export_table(scope_key, df_mat, "redundancia_matriz", "Matriz de Similaridade entre UCs")

    # Pares redundantes
    st.markdown("### 🔗 Pares com Alta Similaridade")
    thr = st.slider("Limiar de redundância (similaridade mínima)", 0.50, 0.95, 0.80, 0.05)
    pares = []
    n = S.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= thr:
                pares.append({"UC A": nomes[i], "UC B": nomes[j], "Similaridade": float(S[i, j])})

    if pares:
        df_pares = pd.DataFrame(pares).sort_values("Similaridade", ascending=False)
        st.dataframe(df_pares.head(100), use_container_width=True)
        export_table(scope_key, df_pares, "redundancia_pares", "Pares de UCs Redundantes")

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df_pares["Similaridade"], bins=10, kde=True, color="#3b5bdb", ax=ax)
        ax.set_title("Distribuição das Similaridades (UCs Redundantes)")
        ax.set_xlabel("Similaridade")
        ax.set_ylabel("Frequência")
        show_and_export_fig(scope_key, fig, "redundancia_hist")
        plt.close(fig)
    else:
        st.info("Nenhum par acima do limiar definido.")

    export_zip_button(scope_key)

    st.markdown("---")
    st.subheader("📘 Interpretação")
    st.markdown(
        "- **≥ 0.90**: possível duplicidade.\n"
        "- **0.75–0.90**: alta sobreposição → revisar escopo/ementa.\n"
        "- **0.60–0.75**: proximidade temática (interdisciplinaridade natural).\n"
        "- **< 0.60**: diferenciação adequada.\n"
    )


# ===============================================================
# 🔬 2) Comparação Frase a Frase
# ===============================================================
def run_pair_analysis(df: pd.DataFrame, scope_key: str):
    st.header("🔬 Análise Frase a Frase entre UCs")
    st.caption(
        "Compara duas UCs **linha a linha**, destacando trechos com maior similaridade semântica."
    )

    col_base = find_col(df, "Ementa") or find_col(df, "Objetos de conhecimento")
    if not col_base:
        st.error("Coluna de texto não encontrada.")
        return

    nomes = df["Nome da UC"].dropna().astype(str).unique().tolist()
    if len(nomes) < 2:
        st.info("É necessário pelo menos duas UCs para comparar.")
        return

    uc_a = st.selectbox("📘 UC A", nomes, key="pair_uc_a")
    uc_b = st.selectbox("📗 UC B", [n for n in nomes if n != uc_a], key="pair_uc_b")

    text_a = replace_semicolons(df.loc[df["Nome da UC"] == uc_a, col_base].iloc[0])
    text_b = replace_semicolons(df.loc[df["Nome da UC"] == uc_b, col_base].iloc[0])

    ph_a, ph_b = _split_sentences(text_a), _split_sentences(text_b)
    if not ph_a or not ph_b:
        st.warning("Textos insuficientes para análise.")
        return

    emb_a, emb_b = sbert_embed(ph_a), sbert_embed(ph_b)
    sim = cosine_similarity(emb_a, emb_b)

    rows = []
    for i in range(len(ph_a)):
        j = int(np.argmax(sim[i]))
        rows.append({"Similaridade": float(sim[i, j]), "Trecho A": ph_a[i], "Trecho B": ph_b[j]})
    df_out = pd.DataFrame(rows).sort_values("Similaridade", ascending=False)

    st.markdown("### 🧩 Trechos mais semelhantes")
    st.dataframe(df_out.head(20).style.format({"Similaridade": "{:.3f}"}), use_container_width=True)
    export_table(scope_key, df_out, f"redundancia_{uc_a}_vs_{uc_b}", f"Frase a Frase — {uc_a} vs {uc_b}")
    export_zip_button(scope_key)

    st.markdown("---")
    st.subheader("📘 Interpretação")
    st.markdown(
        "- **≥ 0.85**: repetição literal.\n"
        "- **0.65–0.85**: paráfrase conceitual.\n"
        "- **< 0.65**: relação periférica.\n"
    )


# ===============================================================
# 🧭 3) Matriz Integrada — Objetos × Egresso × DCN × Competências
# ===============================================================
def run_alignment_matrix(df: pd.DataFrame, scope_key: str, client=None):
    st.header("🧭 Matriz Integrada — Objetos × Egresso × DCN × Competências")
    st.caption(
        """
        Linhas = UCs  
        Colunas = Similaridade entre **Objetos de Conhecimento** e as dimensões curriculares:  
        - Perfil do Egresso  
        - Relação DCN  
        - Competências  

        Quanto mais próximo de **1.00 (verde)**, maior a coerência; valores próximos de **0.00 (vermelho)** indicam desalinhamento.
        """
    )

    col_obj = find_col(df, "Objetos de conhecimento")
    col_egresso = find_col(df, "Competências do Perfil do Egresso")
    col_dcn = find_col(df, "Competências DCN")
    col_comp = find_col(df, "Competências")

    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' não encontrada.")
        return
    if not (col_egresso or col_dcn or col_comp):
        st.error("Nenhuma coluna de competências encontrada (Egresso, DCN ou Competências).")
        return

    df_valid = df.fillna("")
    nomes = df_valid["Nome da UC"].astype(str).tolist()

    modo = st.radio(
        "Selecione o tipo de cálculo de similaridade:",
        ["📊 Por Média da UC", "🧩 Frase a Frase"],
        horizontal=True,
        key="align_mode_radio_v3"
    )

    # ---------------------- Modo 1: Média da UC ----------------------
    if modo == "📊 Por Média da UC":
        textos_obj = df_valid[col_obj].astype(str).apply(replace_semicolons).tolist()
        emb_obj = l2_normalize(sbert_embed(textos_obj))

        results = {"UC": nomes}
        if col_egresso:
            emb_egr = l2_normalize(sbert_embed(df_valid[col_egresso].astype(str).tolist()))
            results["Similaridade Perfil do Egresso"] = np.diag(np.dot(emb_obj, emb_egr.T))
        if col_dcn:
            emb_dcn = l2_normalize(sbert_embed(df_valid[col_dcn].astype(str).tolist()))
            results["Similaridade Relação DCN"] = np.diag(np.dot(emb_obj, emb_dcn.T))
        if col_comp:
            emb_comp = l2_normalize(sbert_embed(df_valid[col_comp].astype(str).tolist()))
            results["Similaridade Competências"] = np.diag(np.dot(emb_obj, emb_comp.T))
        df_res = pd.DataFrame(results)

    # ---------------------- Modo 2: Frase a Frase ----------------------
    else:
        rows = []
        for _, row in df_valid.iterrows():
            nome = str(row["Nome da UC"])
            obj_text = replace_semicolons(str(row.get(col_obj, "")))
            egr_text = replace_semicolons(str(row.get(col_egresso, "")))
            dcn_text = replace_semicolons(str(row.get(col_dcn, "")))
            comp_text = replace_semicolons(str(row.get(col_comp, "")))

            emb_obj = sbert_embed(_split_sentences(obj_text))
            sim_egr = sim_dcn = sim_comp = np.nan

            if egr_text.strip():
                emb_egr = sbert_embed(_split_sentences(egr_text))
                sim_egr = float(np.mean(cosine_similarity(emb_obj, emb_egr)))
            if dcn_text.strip():
                emb_dcn = sbert_embed(_split_sentences(dcn_text))
                sim_dcn = float(np.mean(cosine_similarity(emb_obj, emb_dcn)))
            if comp_text.strip():
                emb_comp = sbert_embed(_split_sentences(comp_text))
                sim_comp = float(np.mean(cosine_similarity(emb_obj, emb_comp)))

            rows.append({
                "UC": nome,
                "Similaridade Perfil do Egresso": sim_egr,
                "Similaridade Relação DCN": sim_dcn,
                "Similaridade Competências": sim_comp
            })
        df_res = pd.DataFrame(rows)

    # Exibir e exportar
    st.markdown("### 📊 Matriz Integrada de Similaridade")
    st.dataframe(safe_style(df_res, cmap="RdYlGn"), use_container_width=True)
    export_table(scope_key, df_res, "matriz_alinhamento_completa", "Matriz Integrada Objetos × Egresso × DCN × Competências")

    # Heatmap
    fig, ax = plt.subplots(figsize=(9, 5))
    df_plot = df_res.set_index("UC")
    for c in df_plot.columns:
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
    sns.heatmap(df_plot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Mapa de Similaridade Curricular Integrado", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, use_container_width=True)

    # Relatório via GPT
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
            "modo": modo,
            "media_egresso": float(medias.get("Similaridade Perfil do Egresso", np.nan)),
            "media_dcn": float(medias.get("Similaridade Relação DCN", np.nan)),
            "media_comp": float(medias.get("Similaridade Competências", np.nan)),
            "ucs_baixas": ucs_baixas,
        }

        prompt = f"""
        Você é um avaliador curricular. Analise os seguintes dados:
        {resumo}

        Gere um relatório técnico breve (até 150 palavras) com:
        - Pontos fortes
        - Fragilidades
        - Recomendações práticas
        Linguagem objetiva, voltada à melhoria curricular.
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
