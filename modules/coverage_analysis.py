import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st

@st.cache_resource
def load_sbert():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def coverage_analysis(df, col_uc="NOME UC", col_comp="COMPETENCIAS", col_ementa="EMENTA"):
    st.header("🧩 Cobertura Curricular por Competência (Mapeamento 1:N)")
    if col_comp not in df.columns or col_ementa not in df.columns:
        st.warning("A base deve conter as colunas 'COMPETENCIAS' e 'EMENTA'.")
        return

    model = load_sbert()
    comp_texts = df[col_comp].dropna().unique().tolist()
    uc_texts = df[col_ementa].tolist()

    emb_comp = model.encode(comp_texts, convert_to_tensor=True)
    emb_uc = model.encode(uc_texts, convert_to_tensor=True)
    sim = util.cos_sim(emb_comp, emb_uc).cpu().numpy()

    coverage = (sim > 0.7).sum(axis=1)
    df_cov = pd.DataFrame({
        "Competência": comp_texts,
        "UCs Relacionadas": coverage,
        "Cobertura (%)": (coverage / len(uc_texts) * 100).round(2)
    }).sort_values("Cobertura (%)", ascending=False)

    st.dataframe(df_cov, use_container_width=True)
    st.bar_chart(df_cov.set_index("Competência")["Cobertura (%)"])
    st.success("Análise de cobertura concluída.")
    return df_cov
