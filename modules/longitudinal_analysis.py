# ===============================================================
# 📈 EmentaLabv2 — Análise Longitudinal (v1.0)
# ===============================================================
# - Analisa evolução semântica de ementas ao longo do tempo
# - Mede mudanças no conteúdo (semelhança SBERT)
# - Detecta UCs com maior alteração entre versões
# - Gera gráficos e relatório interpretativo automático (GPT)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI

from utils.text_utils import find_col, truncate
from utils.embeddings import sbert_embed, l2_normalize
from utils.exportkit import export_table, export_zip_button


# ---------------------------------------------------------------
# 🚀 Função principal
# ---------------------------------------------------------------
def run_longitudinal(df, scope_key, client=None):
    st.header("📈 Análise Longitudinal")
    st.caption(
        "Avalia a **evolução semântica e textual** das ementas ou descrições das UCs ao longo do tempo, "
        "identificando mudanças, estabilidade conceitual e possíveis rupturas entre versões curriculares."
    )

    # -----------------------------------------------------------
    # 🧱 Identificação de colunas
    # -----------------------------------------------------------
    col_text = (
        find_col(df, "Ementa")
        or find_col(df, "Descrição")
        or find_col(df, "Objetos de conhecimento")
    )
    col_uc = find_col(df, "Nome da UC")
    col_periodo = find_col(df, "Período") or find_col(df, "Ano") or find_col(df, "Versão")

    if not (col_text and col_uc and col_periodo):
        st.error("É necessário conter colunas 'Nome da UC', 'Ementa' e 'Período/Ano/Versão' para esta análise.")
        return

    df_valid = df[[col_uc, col_text, col_periodo]].dropna().rename(columns={
        col_uc: "UC",
        col_text: "Texto",
        col_periodo: "Periodo"
    })

    if df_valid.empty:
        st.warning("Nenhuma UC com informações válidas para análise longitudinal.")
        return

    # Normaliza períodos (garantindo ordem)
    df_valid["Periodo"] = df_valid["Periodo"].astype(str).str.strip()
    periodos_unicos = sorted(df_valid["Periodo"].unique().tolist())
    st.info(f"Períodos detectados: {', '.join(periodos_unicos)}")

    # -----------------------------------------------------------
    # 📊 Seleção de UCs e períodos
    # -----------------------------------------------------------
    uc_list = sorted(df_valid["UC"].unique().tolist())
    uc_sel = st.selectbox("Selecione uma UC para análise longitudinal:", uc_list)

    subset = df_valid[df_valid["UC"] == uc_sel].sort_values("Periodo")
    if len(subset) < 2:
        st.warning("É necessário que a UC tenha pelo menos duas versões (em períodos diferentes).")
        return

    # -----------------------------------------------------------
    # 🧠 Cálculo de similaridade semântica entre versões
    # -----------------------------------------------------------
    st.markdown("### 🧠 Evolução Semântica das Versões")
    textos = subset["Texto"].astype(str).tolist()
    emb = l2_normalize(sbert_embed(textos))
    sims = np.dot(emb, emb.T)

    df_sims = pd.DataFrame(sims, index=subset["Periodo"], columns=subset["Periodo"])
    st.dataframe(df_sims.style.format("{:.2f}"), use_container_width=True)
    export_table(scope_key, df_sims, "similaridade_longitudinal", f"Similaridade entre versões da UC {uc_sel}")

    # -----------------------------------------------------------
    # 📉 Visualização — Linha temporal
    # -----------------------------------------------------------
    st.markdown("### 📉 Linha de Similaridade Longitudinal")
    valores = []
    periodos = subset["Periodo"].tolist()
    for i in range(len(periodos) - 1):
        valores.append({
            "Comparação": f"{periodos[i]} → {periodos[i+1]}",
            "Similaridade": float(sims[i, i+1])
        })

    df_line = pd.DataFrame(valores)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=df_line, x="Comparação", y="Similaridade", marker="o", color="#1976D2", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Evolução de Similaridade — {uc_sel}")
    ax.set_ylabel("Similaridade Semântica (0–1)")
    st.pyplot(fig, use_container_width=True)

    # -----------------------------------------------------------
    # ⚠️ Identificação de mudanças relevantes
    # -----------------------------------------------------------
    df_line["Mudança"] = df_line["Similaridade"].apply(
        lambda x: "Ruptura significativa" if x < 0.60 else "Mudança moderada" if x < 0.80 else "Estabilidade"
    )

    st.markdown("### ⚠️ Detecção de Mudanças")
    st.dataframe(df_line, use_container_width=True, hide_index=True)
    export_table(scope_key, df_line, "mudancas_longitudinais", f"Mudanças na UC {uc_sel}")

    rupturas = df_line[df_line["Mudança"] != "Estabilidade"]

    # -----------------------------------------------------------
    # 🧾 Relatório analítico automático (GPT)
    # -----------------------------------------------------------
    if client is None:
        api_key = st.session_state.get("global_api_key", "")
        if api_key:
            client = OpenAI(api_key=api_key)

    if client is not None:
        resumo = (
            f"UC analisada: {uc_sel}\n"
            f"Períodos: {', '.join(periodos)}\n"
            f"Similaridades médias: {df_line['Similaridade'].mean():.2f}\n"
            f"Rupturas detectadas: {len(rupturas)}\n"
        )

        prompt_relatorio = (
            "Você é um especialista em análise curricular. "
            "Com base nas similaridades entre versões da UC a seguir, produza um **relatório breve, objetivo e técnico**, "
            "indicando se a evolução da ementa demonstra **consistência, evolução ou ruptura temática**, "
            "e sugerindo ações de melhoria se necessário.\n\n"
            f"{resumo}"
        )

        try:
            with st.spinner("📄 Gerando relatório analítico via GPT..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt_relatorio}],
                    temperature=0.2,
                )
            analise_gpt = resp.choices[0].message.content.strip()
            st.markdown("### 🧾 Relatório Analítico (Gerado pelo GPT)")
            st.info(analise_gpt)
        except Exception as e:
            st.error(f"❌ Erro ao gerar relatório GPT: {e}")

    # -----------------------------------------------------------
    # 🧭 Interpretação
    # -----------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        ## 🧭 Como interpretar os resultados
        - **Similaridade ≥ 0.85:** estabilidade conceitual — a UC mantém coerência entre versões.  
        - **0.60 ≤ Similaridade < 0.85:** atualização moderada — ajustes textuais e temáticos naturais.  
        - **Similaridade < 0.60:** ruptura temática — revisão substancial no conteúdo.  

        ### 🧩 Aplicações práticas
        - Detectar **mudanças de foco curricular** ou **reformulações significativas**.  
        - Garantir **continuidade evolutiva** nas revisões de PPC.  
        - Identificar **gaps de coerência longitudinal** entre versões sucessivas da mesma UC.  
        """
    )

    export_zip_button(scope_key)
