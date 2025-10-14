# ===============================================================
# 🧠 EmentaLabv2 — Mapa de Bloom (Heurística + GPT Refinement)
# ===============================================================
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from utils.text_utils import find_col
from utils.bloom_helpers import calculate_bloom_level
from utils.exportkit import export_table, show_and_export_fig, export_zip_button

# ---------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------
def run_bloom(df, scope_key):
    # -----------------------------------------------------------
    # 🏷️ Título e descrição
    # -----------------------------------------------------------
    st.header("🧠 Mapa de Bloom — Heurística + GPT")
    st.caption(
        """
        Este módulo analisa os **níveis cognitivos da Taxonomia de Bloom** expressos nos
        *Objetivos de Aprendizagem* das Unidades Curriculares (UCs).

        A classificação ocorre em duas etapas:
        1️⃣ **Heurística automática**, baseada em verbos típicos associados aos níveis de Bloom;  
        2️⃣ **Refinamento GPT (opcional)**, que interpreta semanticamente o texto para ajustar o nível cognitivo.
        """
    )

    # -----------------------------------------------------------
    # 📂 Verificação da coluna base
    # -----------------------------------------------------------
    col_obj = find_col(df, "Objetivo de aprendizagem")
    if not col_obj:
        st.error("Coluna 'Objetivo de aprendizagem' não encontrada.")
        st.stop()

    # -----------------------------------------------------------
    # 🧩 Etapa 1 — Análise Heurística
    # -----------------------------------------------------------
    st.subheader("📊 Distribuição Heurística dos Níveis de Bloom")

    df_out = calculate_bloom_level(df, col_obj)

    freq = df_out["Nível Bloom Predominante"].value_counts(normalize=True).mul(100).round(1)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=freq.index, y=freq.values, ax=ax, palette="crest")
    ax.set_ylabel("% de UCs")
    ax.set_xlabel("Nível de Bloom")
    ax.set_title("Distribuição Heurística de Bloom (verbos e expressões)")
    show_and_export_fig(scope_key, fig, "bloom_distribuicao_heuristica")

    st.dataframe(df_out, use_container_width=True)
    export_table(scope_key, df_out, "bloom_tabela_heuristica", "Bloom Heurístico")

    # -----------------------------------------------------------
    # 🧠 Etapa 2 — Refinamento GPT (opcional)
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Refinamento Inteligente com GPT (opcional)")
    api_key = st.text_input("🔑 OpenAI API Key (opcional para refinamento GPT)", type="password")

    if api_key:
        client = OpenAI(api_key=api_key)
        st.info("O modelo GPT analisará cada objetivo e sugerirá um nível de Bloom mais preciso.")

        # ✅ Corrigido: merge entre df e df_out para garantir a coluna de texto original
        subset = df[["Nome da UC", col_obj]].merge(
            df_out[["Nome da UC", "Nível Bloom Predominante"]],
            on="Nome da UC",
            how="inner"
        ).dropna(subset=[col_obj])

        refined_levels = []
        total = len(subset)

        # Spinner temporário (oculta após o processamento)
        with st.spinner("🧠 Analisando objetivos com GPT..."):
            progress_bar = st.progress(0)

            for i in range(len(subset)):
                objetivo_texto = subset.iloc[i][col_obj]
                nivel_heuristico = subset.iloc[i]["Nível Bloom Predominante"]

                prompt = f"""
                Você é um especialista em taxonomia de Bloom.
                Classifique o seguinte objetivo de aprendizagem no nível cognitivo mais adequado
                (Lembrar, Compreender, Aplicar, Analisar, Avaliar ou Criar)
                e indique o verbo principal usado.

                Objetivo: "{objetivo_texto}"
                Classificação heurística prévia: "{nivel_heuristico}"

                Responda em formato JSON:
                {{
                    "verbo": "...",
                    "nivel_bloom": "...",
                    "justificativa": "..."
                }}
                """

                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    content = resp.choices[0].message.content.strip()
                    refined_levels.append(content)
                except Exception as e:
                    refined_levels.append(f'{{"erro": "{str(e)}"}}')

                progress_bar.progress((i + 1) / total)

        # -------------------------------------------------------
        # 📊 Processamento da saída GPT
        # -------------------------------------------------------
        df_gpt = subset.copy()
        df_gpt["Resultado GPT"] = refined_levels

        # Conversão simplificada (regex de extração)
        df_gpt["Verbo GPT"] = df_gpt["Resultado GPT"].str.extract(r'"verbo"\s*:\s*"([^"]+)"')
        df_gpt["Nível Bloom GPT"] = df_gpt["Resultado GPT"].str.extract(r'"nivel_bloom"\s*:\s*"([^"]+)"')
        df_gpt["Justificativa"] = df_gpt["Resultado GPT"].str.extract(r'"justificativa"\s*:\s*"([^"]+)"')

        # -------------------------------------------------------
        # 📊 Comparativo Heurística × GPT
        # -------------------------------------------------------
        st.markdown("### 📈 Comparativo Heurístico × GPT")
        freq_gpt = df_gpt["Nível Bloom GPT"].value_counts(normalize=True).mul(100).round(1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.barplot(x=freq.index, y=freq.values, ax=ax[0], palette="crest")
        ax[0].set_title("Distribuição Heurística")
        ax[0].set_ylabel("% de UCs")
        ax[0].set_xlabel("Nível de Bloom")

        sns.barplot(x=freq_gpt.index, y=freq_gpt.values, ax=ax[1], palette="rocket")
        ax[1].set_title("Distribuição GPT")
        ax[1].set_ylabel("% de UCs")
        ax[1].set_xlabel("Nível de Bloom (GPT)")

        st.pyplot(fig, use_container_width=True)
        show_and_export_fig(scope_key, fig, "bloom_comparativo_gpt")

        # -------------------------------------------------------
        # 📋 Tabela detalhada e métricas
        # -------------------------------------------------------
        st.markdown("### 📋 Resultados Detalhados por UC")
        df_gpt["Concordância"] = df_gpt.apply(
            lambda r: "✅" if str(r["Nível Bloom GPT"]).strip().lower() == str(r["Nível Bloom Predominante"]).strip().lower() else "⚠️", axis=1
        )

        concord_rate = (df_gpt["Concordância"] == "✅").mean() * 100
        st.metric("Taxa de Concordância Heurística × GPT", f"{concord_rate:.1f}%")

        st.dataframe(
            df_gpt[
                ["Nome da UC", col_obj, "Nível Bloom Predominante", "Nível Bloom GPT", "Verbo GPT", "Concordância", "Justificativa"]
            ],
            use_container_width=True,
        )

        export_table(scope_key, df_gpt, "bloom_refinamento_gpt", "Bloom GPT Refinado")
        export_zip_button(scope_key)

    else:
        st.info("Insira sua chave de API da OpenAI para ativar o refinamento GPT.")
        export_zip_button(scope_key)

    # -----------------------------------------------------------
    # 📘 Interpretação e guia de leitura
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("📘 Como interpretar os resultados")
    st.markdown(
        """
        **1️⃣ Interpretação dos níveis Bloom:**
        - 🧠 *Lembrar*: recordação de informações e fatos básicos.  
        - 💡 *Compreender*: interpretação e explicação de conceitos.  
        - 🧩 *Aplicar*: uso de métodos e conhecimentos em situações práticas.  
        - 🔍 *Analisar*: decomposição e identificação de relações.  
        - ⚖️ *Avaliar*: julgamento crítico e argumentação de decisões.  
        - 🎨 *Criar*: síntese e produção de novas ideias ou artefatos.

        **2️⃣ Uso combinado Heurístico + GPT:**
        - A heurística fornece uma **visão quantitativa rápida** baseada em verbos.  
        - O GPT refina o contexto, considerando **significado semântico e objetivo pedagógico**.  
        - Divergências indicam possíveis **inconsistências na formulação dos objetivos**.

        **3️⃣ Aplicações práticas:**
        - Revisar a coerência entre os objetivos e os níveis cognitivos esperados.  
        - Padronizar a linguagem pedagógica entre cursos ou núcleos.  
        - Subsidiar revisões de PPC e de planos de ensino, fortalecendo evidências de coerência curricular.
        """
    )
