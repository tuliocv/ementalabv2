# ===============================================================
# 🔎 EmentaLabv2 — Relatório de Cobertura
# ===============================================================
import streamlit as st
import pandas as pd
from utils.exportkit import export_table, export_zip_button
from utils.text_utils import find_col

def run_coverage(df, scope_key):
    st.header("🔎 Relatório de Cobertura Curricular")
    st.markdown("""
**O que é:** checagem de preenchimento dos campos críticos (Ementa, Objetos, Competências, Objetivos, Relação DCN).  
**Como analisar:** linhas com **FALTA** inviabilizam parte das análises.
    """)

    cols_chk = {
        "Ementa": find_col(df, "Ementa"),
        "Objetos": find_col(df, "Objetos de conhecimento"),
        "Competências": find_col(df, "Competências do Perfil do Egresso"),
        "Objetivos": find_col(df, "Objetivo de aprendizagem"),
        "Relação DCN": find_col(df, "Relação competência DCN")
    }

    rep = []
    for _, r in df.iterrows():
        linha = {"Nome da UC": r.get("Nome da UC", "—")}
        for nome, col in cols_chk.items():
            if col and col in df.columns:
                v = str(r.get(col, "")).strip()
                linha[nome] = "OK" if v not in ["", "NSA", "N/A", "NULL", "-"] else "FALTA"
            else:
                linha[nome] = "—"
        rep.append(linha)

    df_cov = pd.DataFrame(rep)
    st.dataframe(
        df_cov.style.applymap(lambda v: "background-color:#fef3c7" if v=="FALTA"
                                     else ("background-color:#dcfce7" if v=="OK" else "")),
        use_container_width=True
    )
    export_table(scope_key, df_cov, "relatorio_cobertura", "Cobertura")
    export_zip_button(scope_key)
