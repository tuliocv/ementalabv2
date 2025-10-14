# ===============================================================
# 💬 EmentaLabv2 — Clareza e Sentimento (v5.3)
# ===============================================================
# - Avalia clareza textual e sentimento das ementas/objetivos
# - Compatível com arquitetura do app (scope_key, exportkit)
# - Usa GPT se disponível; fallback SBERT básico se não
# ===============================================================

import streamlit as st
import pandas as pd
from openai import OpenAI

from utils.text_utils import find_col, truncate
from utils.embeddings import sbert_embed, l2_normalize
from utils.exportkit import export_table, export_zip_button


# ---------------------------------------------------------------
# 🚀 Função principal
# ---------------------------------------------------------------
def run_sentiment(df, scope_key, client=None):
    st.header("💬 Clareza e Sentimento das Ementas")
    st.caption(
        "Analisa a clareza textual e o sentimento geral das descrições de cada UC. "
        "Útil para avaliar o tom e a consistência comunicacional das ementas."
    )

    col_text = (
        find_col(df, "Ementa")
        or find_col(df, "Descrição")
        or find_col(df, "Objetos de conhecimento")
    )

    if not col_text:
        st.error("Coluna de texto ('Ementa', 'Descrição' ou 'Objetos de conhecimento') não encontrada.")
        return

    subset = df[["Nome da UC", col_text]].dropna().rename(columns={col_text: "Texto"})
    if subset.empty:
        st.warning("Nenhuma UC com texto preenchido.")
        return

    max_uc = st.slider("Quantidade de UCs (amostra para análise)", 4, min(30, len(subset)), min(12, len(subset)), 1)
    subset = subset.head(max_uc)

    api_key = st.session_state.get("global_api_key", "")
    if api_key:
        client = OpenAI(api_key=api_key)

    results = []

    # -----------------------------------------------------------
    # 🧠 GPT: análise semântica detalhada
    # -----------------------------------------------------------
    if client is not None:
        with st.spinner("🧠 Analisando clareza e sentimento via GPT..."):
            for _, row in subset.iterrows():
                uc_name = row["Nome da UC"]
                text = truncate(str(row["Texto"]), 800)
                prompt = (
                    f"Avalie o texto a seguir em 3 dimensões:\n"
                    f"1️⃣ Clareza textual (Alta, Média, Baixa)\n"
                    f"2️⃣ Sentimento predominante (Positivo, Neutro, Negativo)\n"
                    f"3️⃣ Sugestão de reescrita mais clara e objetiva (máx. 1 frase)\n\n"
                    f"UC: {uc_name}\nTexto:\n{text}"
                )
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    analysis = (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    analysis = f"Erro GPT: {e}"
                results.append([uc_name, text, analysis])
    else:
        # -------------------------------------------------------
        # 🔁 Fallback SBERT: análise aproximada de polaridade
        # -------------------------------------------------------
        st.warning("⚙️ Sem API GPT — aplicando análise SBERT simplificada.")
        emb = l2_normalize(sbert_embed(subset["Texto"].tolist()))
        clarity_scores = emb.std(axis=1)
        sentiment_scores = emb.mean(axis=1)

        for i, r in subset.iterrows():
            clarity = "Alta" if clarity_scores[i] < 0.08 else "Média" if clarity_scores[i] < 0.12 else "Baixa"
            sentiment = "Positivo" if sentiment_scores[i] > 0.05 else "Negativo" if sentiment_scores[i] < -0.05 else "Neutro"
            results.append([r["Nome da UC"], r["Texto"], f"Clareza: {clarity}; Sentimento: {sentiment}"])

    # -----------------------------------------------------------
    # 📊 Exibição e exportação
    # -----------------------------------------------------------
    df_results = pd.DataFrame(results, columns=["Nome da UC", "Texto Original", "Análise"])
    st.markdown("### 📊 Resultados da Análise")
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    export_table(scope_key, df_results, "analise_clareza_sentimento", "Clareza e Sentimento das Ementas")
    export_zip_button(scope_key)
