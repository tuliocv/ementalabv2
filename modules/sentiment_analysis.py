# ===============================================================
# 💬 EmentaLabv2 — Clareza e Sentimento (v5.7 — com Relatório Analítico GPT)
# ===============================================================
# - Analisa clareza textual e sentimento das ementas/objetivos
# - Gera gráficos de distribuição
# - Identifica UCs que requerem intervenção
# - Produz relatório analítico automático via GPT
# ===============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        "Analisa a **clareza textual** e o **sentimento predominante** das ementas ou descrições das UCs. "
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
    if api_key and client is None:
        client = OpenAI(api_key=api_key)

    results = []

    # -----------------------------------------------------------
    # 🧠 Análise via GPT (com estrutura padronizada)
    # -----------------------------------------------------------
    if client is not None:
        with st.spinner("🧠 Analisando clareza e sentimento via GPT..."):
            for _, row in subset.iterrows():
                uc_name = row["Nome da UC"]
                text = truncate(str(row["Texto"]), 800)
                prompt = (
                    f"Avalie o texto a seguir e retorne no formato abaixo:\n\n"
                    f"1️⃣ Clareza textual: Alta / Média / Baixa\n"
                    f"2️⃣ Sentimento predominante: Positivo / Neutro / Negativo\n"
                    f"3️⃣ Sugestão de reescrita mais clara (1 frase no máximo)\n\n"
                    f"UC: {uc_name}\nTexto:\n{text}"
                )
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    analysis = (resp.choices[0].message.content or "").strip()
                    clarity = next((w for w in ["Alta", "Média", "Baixa"] if w.lower() in analysis.lower()), "Indefinida")
                    sentiment = next((w for w in ["Positivo", "Neutro", "Negativo"] if w.lower() in analysis.lower()), "Indefinido")
                except Exception as e:
                    analysis = f"Erro GPT: {e}"
                    clarity, sentiment = "Indefinida", "Indefinido"

                results.append([uc_name, text, clarity, sentiment, analysis])
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
            results.append([r["Nome da UC"], r["Texto"], clarity, sentiment, analysis])

    # -----------------------------------------------------------
    # 📊 Resultados consolidados
    # -----------------------------------------------------------
    df_results = pd.DataFrame(results, columns=["Nome da UC", "Texto Original", "Clareza", "Sentimento", "Análise"])
    st.markdown("### 📊 Resultados da Análise")
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    export_table(scope_key, df_results, "analise_clareza_sentimento", "Clareza e Sentimento das Ementas")

    # -----------------------------------------------------------
    # 📈 Gráficos de distribuição
    # -----------------------------------------------------------
    st.markdown("### 📈 Distribuição de Clareza e Sentimento")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(data=df_results, x="Clareza", palette="Blues", order=["Alta", "Média", "Baixa"], ax=ax)
        ax.set_title("Distribuição de Clareza")
        ax.set_xlabel("")
        ax.set_ylabel("Quantidade de UCs")
        st.pyplot(fig, use_container_width=True)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.countplot(data=df_results, x="Sentimento", palette="coolwarm", order=["Positivo", "Neutro", "Negativo"], ax=ax2)
        ax2.set_title("Distribuição de Sentimento")
        ax2.set_xlabel("")
        ax2.set_ylabel("Quantidade de UCs")
        st.pyplot(fig2, use_container_width=True)

    # -----------------------------------------------------------
    # ⚠️ Relatório de UCs que requerem intervenção
    # -----------------------------------------------------------
    df_intervencao = df_results[
        (df_results["Clareza"].str.contains("Baixa", case=False, na=False))
        | (df_results["Sentimento"].str.contains("Negativo", case=False, na=False))
    ]

    st.markdown("### ⚠️ UCs que Requerem Intervenção")
    if df_intervencao.empty:
        st.success("✅ Nenhuma UC requer revisão imediata. Os textos apresentam boa clareza e tom adequado.")
    else:
        st.warning(
            f"⚠️ Foram identificadas **{len(df_intervencao)} UCs** com baixa clareza e/ou tom negativo. "
            "Recomenda-se revisão dos textos dessas ementas."
        )
        st.dataframe(df_intervencao[["Nome da UC", "Clareza", "Sentimento", "Análise"]], use_container_width=True, hide_index=True)
        export_table(scope_key, df_intervencao, "ucs_revisao_textual", "UCs que Requerem Intervenção")

    # -----------------------------------------------------------
    # 🧾 Relatório Analítico Automático (via GPT)
    # -----------------------------------------------------------
    if client is not None:
        st.markdown("### 🧾 Relatório Analítico da Estrutura Curricular (Gerado pelo GPT)")
        try:
            resumo = "Resumo dos resultados:\n"
            resumo += f"- Total de UCs analisadas: {len(df_results)}\n"
            resumo += f"- UCs com clareza baixa: {len(df_results[df_results['Clareza'].str.contains('Baixa', case=False, na=False)])}\n"
            resumo += f"- UCs com tom negativo: {len(df_results[df_results['Sentimento'].str.contains('Negativo', case=False, na=False)])}\n\n"
            resumo += "Lista das UCs críticas:\n"
            for uc in df_intervencao["Nome da UC"].tolist()[:10]:
                resumo += f"- {uc}\n"

            prompt_relatorio = (
                "Você é um avaliador de currículos acadêmicos. "
                "Com base nos dados a seguir, elabore um **relatório breve, direto e objetivo** sobre os resultados da análise de clareza e sentimento, "
                "destacando **pontos fortes**, **fragilidades** e **recomendações práticas** de melhoria.\n\n"
                f"{resumo}"
            )

            with st.spinner("📄 Gerando relatório analítico via GPT..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_relatorio}],
                    temperature=0.2,
                )
            analise_gpt = resp.choices[0].message.content.strip()
            st.info(analise_gpt)
        except Exception as e:
            st.error(f"❌ Erro ao gerar relatório analítico GPT: {e}")

    # -----------------------------------------------------------
    # 🧭 Interpretação e aplicação
    # -----------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        ## 🧭 Interpretação dos Resultados
        ### 📘 O que esta análise realiza
        Avalia a **clareza linguística** e o **tom emocional** das ementas, identificando textos que possam gerar dúvidas, redundâncias ou impacto negativo na percepção institucional.

        ### 🔎 Como interpretar
        - **Clareza Alta:** linguagem direta e estruturada.  
        - **Clareza Média:** compreensível, mas pode ser simplificada.  
        - **Clareza Baixa:** confusa, prolixa ou genérica — requer revisão.  
        - **Sentimento Positivo:** texto construtivo e inspirador.  
        - **Sentimento Neutro:** adequado a contextos técnicos.  
        - **Sentimento Negativo:** pode transmitir insegurança ou rigidez.

        ### 🧩 Ações recomendadas
        - Revisar UCs com **clareza baixa** ou **tom negativo**.  
        - Promover **padronização textual** entre cursos.  
        - Garantir que os textos reflitam **coerência institucional** e **comunicação inclusiva**.
        """
    )

    export_zip_button(scope_key)
