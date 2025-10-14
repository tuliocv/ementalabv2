# ===============================================================
# 🧬 EmentaLabv2 — Similaridade, Redundância e Alinhamento (v11.6)
# ===============================================================
# Entradas públicas esperadas pelo app:
#   - run_redundancy(df, scope_key)
#   - run_pair_analysis(df, scope_key)
#   - run_alignment_matrix(df, scope_key, client=None)
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
    OpenAI = None  # opcional

from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, export_zip_button, show_and_export_fig
from utils.text_utils import find_col, replace_semicolons, _split_sentences


# ===============================================================
# 🔧 Estilização segura de DataFrame numérico (0..1)
# ===============================================================
def safe_style(df: pd.DataFrame, cmap="RdYlGn", decimals=2):
    if df.empty:
        return df
    df_fmt = df.copy()
    # Tenta converter todas as colunas (menos a primeira "UC") para número
    num_cols = [c for c in df_fmt.columns if c != "UC"]
    for c in num_cols:
        df_fmt[c] = pd.to_numeric(df_fmt[c], errors="coerce")
    fmt_dict = {c: f"{{:.{decimals}f}}" for c in num_cols}
    return df_fmt.style.format(fmt_dict).background_gradient(cmap=cmap, vmin=0, vmax=1)


# ===============================================================
# 🔁 1) Redundância global entre UCs
# ===============================================================
def run_redundancy(df: pd.DataFrame, scope_key: str):
    st.header("🧬 Redundância entre UCs")
    st.caption(
        "Compara ementas/objetos de conhecimento via SBERT para detectar "
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

    # Matriz (exibição reduzida na tela)
    st.markdown("### 🧮 Matriz de Similaridade Global (amostra)")
    df_mat = pd.DataFrame(S, index=nomes, columns=nomes)
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

        # Histograma
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
    st.subheader("📘 Como interpretar")
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
        "Compara duas UCs **linha a linha**, destacando trechos com maior similaridade."
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
    st.dataframe(
        df_out.head(20).style.format({"Similaridade": "{:.3f}"}),
        use_container_width=True,
    )
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
# 🧭 3) Matriz Integrada: Objetos × Egresso × DCN
# ===============================================================
def run_alignment_matrix(df: pd.DataFrame, scope_key: str, client=None):
    st.header("🧭 Matriz de Similaridade Integrada — Objetos × Competências & DCN")
    st.caption(
        "Linhas = UCs • Colunas = Similaridade Objetos × Egresso e Objetos × DCN. "
        "Escolha entre cálculo por **média da UC** ou **frase a frase**. "
        "Cores: verde (≈1) = forte coerência; vermelho (≈0) = fraca."
    )

    col_obj = find_col(df, "Objetos de conhecimento")
    col_comp = find_col(df, "Competências do Perfil do Egresso")
    col_dcn = find_col(df, "Competências DCN")
    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' não encontrada.")
        return
    if not (col_comp or col_dcn):
        st.error("Nenhuma coluna de competências encontrada (Egresso/DCN).")
        return

    df_valid = df.fillna("")
    nomes = df_valid["Nome da UC"].astype(str).tolist()

    # Modo de cálculo
    st.markdown("### ⚙️ Modo de Cálculo")
    modo = st.radio(
        "Selecione o tipo de cálculo:",
        ["📊 Por Média da UC", "🧩 Frase a Frase"],
        horizontal=True,
        key="align_mode_radio"
    )

    # ----------------- Modo 1: Média da UC -----------------
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

    # ----------------- Modo 2: Frase a Frase -----------------
    else:
        rows = []
        for _, row in df_valid.iterrows():
            nome = str(row["Nome da UC"])
            obj_text = replace_semicolons(str(row.get(col_obj, "")))
            comp_text = replace_semicolons(str(row.get(col_comp, "")))
            dcn_text = replace_semicolons(str(row.get(col_dcn, "")))

            # Frases
            objs = _split_sentences(obj_text)
            emb_obj = sbert_embed(objs)

            sim_egresso = np.nan
            sim_dcn = np.nan

            if col_comp and comp_text.strip():
                comps = _split_sentences(comp_text)
                emb_comp = sbert_embed(comps)
                if emb_obj.size and emb_comp.size:
                    sim_egresso = float(cosine_similarity(emb_obj, emb_comp).mean())

            if col_dcn and dcn_text.strip():
                dcns = _split_sentences(dcn_text)
                emb_dcn = sbert_embed(dcns)
                if emb_obj.size and emb_dcn.size:
                    sim_dcn = float(cosine_similarity(emb_obj, emb_dcn).mean())

            rows.append({
                "UC": nome,
                "Similaridade (Objetos × Egresso)": sim_egresso,
                "Similaridade (Objetos × DCN)": sim_dcn,
            })
        df_res = pd.DataFrame(rows)

    # Exibição e export
    st.markdown("### 📊 Matriz Integrada")
    st.dataframe(safe_style(df_res, cmap="RdYlGn"), use_container_width=True)
    export_table(scope_key, df_res, "matriz_alinhamento_unificada", "Matriz Integrada de Similaridade")

    # Heatmap
    if not df_res.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        heat = df_res.set_index("UC")
        for c in heat.columns:
            heat[c] = pd.to_numeric(heat[c], errors="coerce")
        sns.heatmap(
            heat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, ax=ax
        )
        ax.set_title("Mapa de Similaridade (Objetos × Egresso × DCN)", fontsize=13, fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        show_and_export_fig(scope_key, fig, "matriz_alinhamento_heatmap")
        plt.close(fig)

    # Relatório GPT (opcional)
    if client is None:
        api_key = st.session_state.get("global_api_key", "")
        if api_key and OpenAI is not None:
            try:
                client = OpenAI(api_key=api_key)
            except Exception:
                client = None

    st.markdown("---")
    st.subheader("🧾 Relatório Analítico de Alinhamento Curricular")
    if client is not None:
        medias = df_res.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").mean()
        ucs_baixas = df_res[df_res.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").mean(axis=1) < 0.65]["UC"].tolist()

        resumo = {
            "modo": "media_uc" if modo.startswith("📊") else "frase_a_frase",
            "media_egresso": float(medias.get("Similaridade (Objetos × Egresso)", np.nan)) if len(medias) else None,
            "media_dcn": float(medias.get("Similaridade (Objetos × DCN)", np.nan)) if len(medias) else None,
            "ucs_baixas": ucs_baixas,
            "n_ucs": int(len(df_res)),
        }

        prompt = (
            "Você é um avaliador curricular. Analise os dados abaixo e produza um relatório curto (≤150 palavras), "
            "com linguagem objetiva, contendo: Pontos fortes, Fragilidades e Recomendações práticas.\n\n"
            f"{resumo}"
        )

        try:
            with st.spinner("🧠 Gerando relatório via GPT..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            analise = (resp.choices[0].message.content or "").strip()
            if analise:
                st.success("Relatório gerado com sucesso.")
                st.markdown(analise)
        except Exception as e:
            st.error(f"Erro ao gerar relatório via GPT: {e}")
    else:
        st.info("🔑 Chave da OpenAI não encontrada — relatório não gerado.")

    export_zip_button(scope_key)
