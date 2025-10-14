# ===============================================================
# 🧩 EmentaLabv2 — Similaridade Objetos × Competências × DCN (v8.2.1)
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, export_zip_button
from utils.text_utils import find_col, replace_semicolons

# ---------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------
def run_similarity(df, scope_key):
    """
    Analisa a similaridade semântica entre:
    - Objetos de conhecimento
    - Competências do perfil do egresso
    - Relação com as competências das DCNs
    usando embeddings SBERT (MiniLM Multilingual).
    """

    st.header("🧩 Similaridade (Objetos × Competências & DCN)")
    st.markdown("""
**Objetivo:** verificar o grau de alinhamento semântico entre o conteúdo trabalhado nas UCs
(objetos de conhecimento) e as competências esperadas (egresso e DCN).

**Interpretação:**  
Valores próximos de **1.0** indicam alto alinhamento semântico.  
Valores próximos de **0.0** indicam baixo alinhamento.
    """)

    # -----------------------------------------------------------
    # 1. Identificação automática das colunas
    # -----------------------------------------------------------
    col_obj = find_col(df, "Objetos de conhecimento")
    col_comp = find_col(df, "Competências do Perfil do Egresso")
    col_dcn = find_col(df, "Relação competência DCN")

    if not all([col_obj, col_comp, col_dcn]):
        st.warning("⚠️ Não foram encontradas todas as colunas necessárias para esta análise.")
        st.markdown(f"""
        - Objetos de conhecimento → `{col_obj or '❌ não encontrado'}`
        - Competências do egresso → `{col_comp or '❌ não encontrado'}`
        - Relação competência DCN → `{col_dcn or '❌ não encontrado'}`
        """)
        st.stop()

    # -----------------------------------------------------------
    # 2. Preparação dos dados
    # -----------------------------------------------------------
    df_an = df[["Nome da UC", col_obj, col_comp, col_dcn]].dropna(subset=[col_obj])
    if df_an.empty:
        st.error("❌ Nenhuma linha válida encontrada para análise (verifique se as colunas estão preenchidas).")
        st.stop()

    objetos = df_an[col_obj].astype(str).apply(replace_semicolons).tolist()
    comp = df_an[col_comp].astype(str).tolist()
    dcn = df_an[col_dcn].astype(str).tolist()
    nomes = df_an["Nome da UC"].astype(str).tolist()

    # -----------------------------------------------------------
    # 3. Geração dos embeddings e cálculo de similaridade
    # -----------------------------------------------------------
    with st.spinner("Gerando embeddings SBERT e calculando similaridades..."):
        emb_obj = l2_normalize(sbert_embed(objetos))
        emb_comp = l2_normalize(sbert_embed(comp))
        emb_dcn = l2_normalize(sbert_embed(dcn))

        sim_obj_comp = np.sum(emb_obj * emb_comp, axis=1)
        sim_obj_dcn = np.sum(emb_obj * emb_dcn, axis=1)

    # -----------------------------------------------------------
    # 4. Montagem do DataFrame de resultados
    # -----------------------------------------------------------
    df_sim = pd.DataFrame({
        "Nome da UC": nomes,
        "Sim. Objetos × Competências": sim_obj_comp,
        "Sim. Objetos × Relação DCN": sim_obj_dcn
    })

    # -----------------------------------------------------------
    # 5. Renderização segura e estilizada
    # -----------------------------------------------------------
    numeric_cols = ["Sim. Objetos × Competências", "Sim. Objetos × Relação DCN"]
    for c in numeric_cols:
        df_sim[c] = pd.to_numeric(df_sim[c], errors="coerce")

    st.subheader("📊 Tabela de Similaridade")
    st.caption("As colunas abaixo representam o grau de similaridade semântica entre as dimensões analisadas.")

    try:
        st.dataframe(
            df_sim.style
            .format({c: "{:.3f}" for c in numeric_cols})
            .background_gradient(cmap="RdYlGn", subset=numeric_cols),
            use_container_width=True
        )
    except Exception:
        # fallback seguro caso Styler falhe (por exemplo, no Streamlit Cloud)
        st.dataframe(df_sim, use_container_width=True)

    # -----------------------------------------------------------
    # 6. Estatísticas gerais
    # -----------------------------------------------------------
    st.markdown("### 📈 Estatísticas gerais")
    col1, col2 = st.columns(2)
    col1.metric("Média Objetos × Competências", f"{df_sim[numeric_cols[0]].mean():.3f}")
    col2.metric("Média Objetos × DCN", f"{df_sim[numeric_cols[1]].mean():.3f}")

    st.markdown("---")
    st.info("""
📌 **Dica:** use esta análise para validar se os *Objetos de conhecimento*
estão semanticamente próximos das *Competências* esperadas.  
Quanto maior a coerência, maior a consistência curricular percebida.
    """)

    # -----------------------------------------------------------
    # 7. Exportação dos resultados
    # -----------------------------------------------------------
    export_table(scope_key, df_sim, "similaridade_obj_comp_dcn", "Similaridade")
    export_zip_button(scope_key)
