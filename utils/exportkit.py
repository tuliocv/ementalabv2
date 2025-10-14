# ===============================================================
# 📦 EmentaLabv2 — Export Helpers (CSV, Excel, Zip, Fig)
# ===============================================================
import streamlit as st
import io, zipfile, base64
import pandas as pd
import matplotlib.pyplot as plt

_exports = {}

def _init_exports(scope_key: str):
    """Inicializa dicionário de exportações por sessão"""
    if scope_key not in _exports:
        _exports[scope_key] = {}

def export_table(scope_key, df, key, label):
    """Adiciona tabela ao pacote ZIP temporário"""
    _exports[scope_key][key] = df
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"💾 Baixar {label} (CSV)", data=csv_bytes, file_name=f"{key}.csv", mime="text/csv")

def show_and_export_fig(scope_key, fig: plt.Figure, key: str):
    """Mostra e prepara exportação de figura"""
    st.pyplot(fig, use_container_width=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    _exports[scope_key][f"{key}.png"] = buf.getvalue()

def export_zip_button(scope_key):
    """Gera botão para baixar todos os arquivos exportados"""
    if not _exports.get(scope_key):
        return
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        for name, obj in _exports[scope_key].items():
            if isinstance(obj, pd.DataFrame):
                z.writestr(f"{name}.csv", obj.to_csv(index=False))
            elif isinstance(obj, (bytes, bytearray)):
                z.writestr(name, obj)
    zip_buf.seek(0)
    st.download_button("📦 Baixar Pacote ZIP Completo", data=zip_buf, file_name=f"{scope_key}.zip", mime="application/zip")

def get_docx_bytes(text: str):
    """Cria um DOCX simples com texto para download"""
    from docx import Document
    doc = Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()
