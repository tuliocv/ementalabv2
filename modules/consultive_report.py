# ===============================================================
# 🧾 EmentaLabv2 — Relatório Consultivo Integrado (v9.1)
# ===============================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from utils.exportkit import export_zip_button, export_table, get_docx_bytes
from utils.text_utils import find_col
import io


# ---------------------------------------------------------------
# 1. Função auxiliar — gráficos de apoio
# ---------------------------------------------------------------
def _plot_bloom_distribution(df, col_bloom: str):
    """Gráfico simples de distribuição dos níveis Bloom."""
    if col_bloom not in df.columns:
        st.warning("Coluna de níveis Bloom não encontrada.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=col_bloom, data=df, order=df[col_bloom].value_counts().index, palette="viridis", ax=ax)
    ax.set_title("Distribuição dos Níveis Bloom nas UCs")
    ax.set_xlabel("Nível Cognitivo")
    ax.set_ylabel("Quantidade de UCs")
    st.pyplot(fig, use_container_width=True)


def _plot_sentiment_bar(df, col_sent: str):
    """Gráfico de barras de sentimento das ementas."""
    if col_sent not in df.columns:
        st.warning("Coluna de sentimento não encontrada.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[col_sent], bins=10, kde=True, color="#3b5bdb", ax=ax)
    ax.set_title("Distribuição de Sentimento (Clareza Linguística)")
    ax.set_xlabel("Polaridade (−1 a +1)")
    ax.set_ylabel("Frequência")
    st.pyplot(fig, use_container_width=True)


# ---------------------------------------------------------------
# 2. Função principal
# ---------------------------------------------------------------
def run_consultive(df: pd.DataFrame, scope_key: str):
    """
    Gera um relatório consultivo consolidando análises de:
    - Cobertura Curricular (Competências DCN × UCs)
    - Curva Bloom Progressiva
    - Convergência Temática
    - Dependência Curricular
    - Clareza e Sentimento
    - Análise Longitudinal
    """

    st.header("🧾 Relatório Consultivo Integrado — EmentaLabv2")
    st.caption("Síntese analítica consolidada das dimensões curriculares avaliadas pelo EmentaLabv2.")

    # -----------------------------------------------------------
    # Parâmetros
    # -----------------------------------------------------------
    api_key = st.text_input("🔑 OpenAI API Key (opcional para gerar texto automático)", type="password")
    gerar_texto = st.checkbox("🧠 Gerar análise automática via GPT", value=False)
    if gerar_texto and not api_key:
        st.warning("Para gerar o texto automático, insira a API Key acima.")

    st.markdown("---")
    st.subheader("📊 Indicadores de Cobertura e Complexidade")

    # -----------------------------------------------------------
    # 1. Cobertura Curricular (competências)
    # -----------------------------------------------------------
    col_comp = find_col(df, "Competência") or find_col(df, "Competências do Perfil do Egresso")
    if col_comp:
        st.markdown("### 🎯 Cobertura de Competências")
        df_cov = df[["Nome da UC", col_comp]].dropna()
        st.write(f"**{len(df_cov)}** UCs possuem competências explicitadas.")
        export_table(scope_key, df_cov, "cobertura_competencias", "Cobertura de Competências")
    else:
        st.info("Nenhuma coluna de competências encontrada no dataset.")

    # -----------------------------------------------------------
    # 2. Curva Bloom
    # -----------------------------------------------------------
    col_bloom = find_col(df, "Nível Bloom") or find_col(df, "Bloom")
    if col_bloom:
        _plot_bloom_distribution(df, col_bloom)
        export_table(scope_key, df[["Nome da UC", col_bloom]], "curva_bloom", "Curva Bloom")
    else:
        st.info("Não há coluna de níveis Bloom gerados.")

    # -----------------------------------------------------------
    # 3. Clareza e Sentimento
    # -----------------------------------------------------------
    st.markdown("### 💬 Clareza e Sentimento das Ementas")
    col_sent = find_col(df, "sentimento") or find_col(df, "polaridade")
    if col_sent:
        _plot_sentiment_bar(df, col_sent)
        mean_sent = df[col_sent].mean()
        st.metric("Média geral de polaridade", f"{mean_sent:.3f}")
        export_table(scope_key, df[["Nome da UC", col_sent]], "sentimento", "Clareza e Sentimento")
    else:
        st.info("Nenhuma análise de sentimento encontrada no dataset.")

    # -----------------------------------------------------------
    # 4. Convergência Temática (clusters)
    # -----------------------------------------------------------
    st.markdown("### 🧩 Convergência Temática (Clusters)")
    if "Cluster" in df.columns:
        st.write(f"Total de **{df['Cluster'].nunique()}** agrupamentos temáticos detectados.")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="Cluster", data=df, palette="crest", ax=ax)
        ax.set_title("Distribuição de UCs por Cluster Temático")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Número de UCs")
        st.pyplot(fig)
        export_table(scope_key, df[["Nome da UC", "Cluster"]], "clusters_tematica", "Clusters Temáticos")
    else:
        st.info("Nenhuma coluna de cluster encontrada.")

    # -----------------------------------------------------------
    # 5. Dependência Curricular (pré-requisitos)
    # -----------------------------------------------------------
    st.markdown("### 🔗 Relações de Dependência Curricular")
    if {"Pré-requisito", "UC Dependente"}.issubset(df.columns):
        rel = df[["Pré-requisito", "UC Dependente"]].drop_duplicates()
        st.write(f"Foram identificadas **{len(rel)}** relações de pré-requisito.")
        export_table(scope_key, rel, "dependencias_curriculares", "Dependências Curriculares")
    else:
        st.info("Nenhuma relação de pré-requisito detectada neste dataset.")

    # -----------------------------------------------------------
    # 6. Análise Longitudinal
    # -----------------------------------------------------------
    st.markdown("### ⏳ Análise Longitudinal de Revisões")
    col_rev = find_col(df, "versão") or find_col(df, "última atualização") or find_col(df, "ano")
    if col_rev:
        anos = df[col_rev].dropna().astype(str)
        if not anos.empty:
            freq = anos.value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            freq.plot(kind="bar", color="#748ffc", ax=ax)
            ax.set_title("Distribuição de UCs por Ano/Versão")
            ax.set_xlabel("Ano / Versão")
            ax.set_ylabel("Número de UCs")
            st.pyplot(fig)
            export_table(scope_key, df[["Nome da UC", col_rev]], "longitudinal", "Análise Longitudinal")
    else:
        st.info("Nenhum campo temporal identificado (ano, versão ou atualização).")

    # -----------------------------------------------------------
    # 7. Síntese automática (GPT opcional)
    # -----------------------------------------------------------
    if gerar_texto and api_key:
        st.markdown("---")
        st.subheader("🧠 Síntese Automática via GPT")

        client = OpenAI(api_key=api_key)
        prompt = (
            "Gere um resumo consultivo de um relatório curricular, considerando as dimensões:\n"
            "- Cobertura de competências\n"
            "- Curva Bloom (níveis cognitivos)\n"
            "- Convergência temática (clusters)\n"
            "- Dependências curriculares\n"
            "- Clareza e sentimento linguístico\n"
            "- Análise longitudinal\n\n"
            "Descreva pontos fortes, lacunas e recomendações, em um tom profissional e institucional."
        )

        with st.spinner("Gerando texto analítico via GPT..."):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
        texto_final = resp.choices[0].message.content.strip()

        st.text_area("📄 Síntese Automática", value=texto_final, height=300)
        st.download_button(
            "💾 Baixar Síntese (TXT)",
            data=texto_final,
            file_name="sintese_consultiva.txt",
            mime="text/plain",
        )

        # Exporta versão DOCX
        docx_bytes = get_docx_bytes(texto_final)
        st.download_button(
            "📄 Baixar Relatório (DOCX)",
            data=docx_bytes,
            file_name="relatorio_consultivo.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    # -----------------------------------------------------------
    # Finalização
    # -----------------------------------------------------------
    st.markdown("---")
    export_zip_button(scope_key)
    st.success("✅ Relatório consultivo gerado com sucesso!")
