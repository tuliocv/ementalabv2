import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import streamlit as st

@st.cache_resource
def load_long_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def longitudinal_analysis(df_antigo, df_novo, col_ementa="EMENTA", col_nome="NOME UC"):
    st.header("🧮 Análise Longitudinal de Revisões Curriculares (Governança 2.0)")

    if col_ementa not in df_antigo.columns or col_ementa not in df_novo.columns:
        st.warning("As bases devem conter a coluna 'EMENTA'.")
        return

    model = load_long_model()
    emb_old = model.encode(df_antigo[col_ementa].tolist(), convert_to_tensor=True)
    emb_new = model.encode(df_novo[col_ementa].tolist(), convert_to_tensor=True)

    sim = util.cos_sim(emb_old, emb_new).cpu().numpy()
    delta = 1 - np.diag(sim)

    df_delta = pd.DataFrame({
        "UC": df_novo[col_nome],
        "Mudança (%)": (delta * 100).round(2)
    })

    st.dataframe(df_delta, use_container_width=True)
    st.bar_chart(df_delta.set_index("UC")["Mudança (%)"])
    st.info("Valores altos indicam maior modificação semântica nas ementas.")
    return df_delta
