# ===============================================================
# 💬 EmentaLabv2 — Clareza e Sentimento (v5.4 — com explicação interpretativa)
# ===============================================================
# - Avalia clareza textual e sentimento das ementas/objetivos
# - Compatível com arquitetura do app (scope_key, exportkit)
# - Usa GPT se disponível; fallback SBERT básico se não
# - Explica o objetivo e como interpretar os resultados
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
        "Analisa a **clareza textual** e o **sentimento predominante** das descrições ou ementas das Unidades Curriculares (UCs). "
        "O objetivo é apoiar a revisão de textos institucionais, observando tom, objetividade e coerência comunicacional."
    )

    # -----------------------------------------------------------
    # 🔍 Identificação da coluna de texto
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # 🔑 Chave e cliente OpenAI
    # -----------------------------------------------------------
    api_key = st.session_state.get("global_api_key", "")
    if api_key:
        client = OpenAI(api_key=api_key)

    results = []

    # -----------------------------------------------------------
    # 🧠 Análise via GPT (com interpretação completa)
    # -----------------------------------------------------------
    if client is not None:
        with st.spinner("🧠 Analisando clareza e sentimento via GPT..."):
            for _, row in subset.iterrows():
                uc_name = row["Nome da UC"]
                text = truncate(str(row["Texto"]), 800)
                prompt = (
                    f"Analise o texto a seguir considerando clareza e tom comunicacional.\n\n"
                    f"Retorne de forma estruturada:\n"
                    f"1️⃣ Clareza textual: Alta / Média / Baixa\n"
                    f"2️⃣ Sentimento predominante: Positivo / Neutro / Negativo\n"
                    f"3️⃣ Sugestão de reescrita mais clara (até 1 frase)\n\n"
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
        # 🔁 Fallback SBERT (sem GPT)
        # -------------------------------------------------------
        st.warning("⚙️ Sem API GPT — aplicando análise SBERT simplificada.")
        emb = l2_normalize(sbert_embed(subset["Texto"].tolist()))
        clarity_scores = emb.std(axis=1)
        sentiment_scores = emb.mean(axis=1)

        for i, r in subset.iterrows():
            clarity = "Alta" if clarity_scores[i] < 0.08 else "Média" if clarity_scores[i] < 0.12 else "Baixa"
            sentiment = "Positivo" if sentiment_scores[i] > 0.05 else "Negativo" if sentiment_scores[i] < -0.05 else "Neutro"
            analysis = f"Clareza: {clarity} | Sentimento: {sentiment}"
            results.append([r["Nome da UC"], r["Texto"], analysis])

    # -----------------------------------------------------------
    # 📊 Exibição dos resultados
    # -----------------------------------------------------------
    df_results = pd.DataFrame(results, columns=["Nome da UC", "Texto Original", "Análise"])
    st.markdown("### 📊 Resultados da Análise")
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------
    # 📈 Síntese de distribuição (opcional)
    # -----------------------------------------------------------
    if client is None:
        st.info("💡 Conecte sua API OpenAI para uma análise de texto mais interpretativa e precisa.")

    # -----------------------------------------------------------
    # 💾 Exportação
    # -----------------------------------------------------------
    export_table(scope_key, df_results, "analise_clareza_sentimento", "Clareza e Sentimento das Ementas")
    export_zip_button(scope_key)

    # -----------------------------------------------------------
    # 🧭 Interpretação e finalidade
    # -----------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        ## 🧭 Interpretação dos Resultados
        ### 📘 O que esta análise realiza
        Esta ferramenta avalia o **nível de clareza** (objetividade, fluidez e precisão linguística) e o **tom emocional** dos textos das ementas.  
        Ela ajuda coordenadores e docentes a identificar:
        - Ementas excessivamente longas, vagas ou redundantes;
        - Textos com linguagem excessivamente técnica ou emocional;
        - Necessidade de padronização entre as UCs de um mesmo curso.

        ### 🔎 Como interpretar
        - **Clareza Alta:** texto direto, coerente e bem estruturado.  
        - **Clareza Média:** texto compreensível, mas com termos genéricos ou frases longas.  
        - **Clareza Baixa:** linguagem confusa, prolixa ou com excesso de jargões.  
        - **Sentimento Positivo:** linguagem motivadora e construtiva.  
        - **Sentimento Neutro:** estilo descritivo, comum em textos técnicos.  
        - **Sentimento Negativo:** tom excessivamente crítico ou ambíguo.

        ### 🧩 Aplicação prática
        Use os resultados para:
        - Revisar ementas que apresentem **baixa clareza** ou tom **negativo**;  
        - Promover maior **consistência editorial** entre as UCs;  
        - Padronizar o estilo de redação institucional para relatórios e PPCs.
        """
    )
