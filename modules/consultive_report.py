# ===============================================================
# 🧾 EmentaLabv2 — Relatório Consultivo Integrado (placeholder)
# ===============================================================
import streamlit as st
import pandas as pd
from utils.exportkit import export_zip_button

def run_consultive(df: pd.DataFrame, scope_key: str):
    """
    Gera um relatório consultivo (resumo consolidado das análises).
    Esta é uma versão placeholder — pode ser substituída depois por
    uma versão que compile todas as métricas de Cobertura, Bloom,
    Convergência, Dependência, Clareza e Longitudinal.
    """
    st.header("🧾 Relatório Consultivo Integrado")
    st.info(
        "Este módulo compila resultados de todas as análises "
        "em um relatório único. Em breve, incluirá gráficos "
        "e síntese automática das métricas principais."
    )

    st.markdown("### 📘 Status")
    st.success("✅ Módulo carregado corretamente (versão placeholder).")

    export_zip_button(scope_key)
