# ===============================================================
# 💾 EmentaLabv2 — ExportKit Utilitário
# ===============================================================
# Responsável por inicializar diretórios temporários de exportação,
# salvar tabelas, gráficos e gerar pacotes .zip de resultados.
# ---------------------------------------------------------------
# Compatível com versões antigas (aceita _init_exports() sem argumento)
# ===============================================================

import os
import io
import tempfile
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime


# ---------------------------------------------------------------
# 🔧 Inicializa diretório temporário de exportação
# ---------------------------------------------------------------
def _init_exports(scope_key: str = "default"):
    """
    Cria (ou reutiliza) um diretório temporário de exportação.
    Compatível com chamadas com ou sem argumento.
    """
    export_dir = os.path.join(tempfile.gettempdir(), f"ementalab_exports_{scope_key}")
    os.makedirs(export_dir, exist_ok=True)
    return export_dir


# ---------------------------------------------------------------
# 💾 Exporta DataFrame como Excel/CSV
# ---------------------------------------------------------------
def export_table(scope_key: str, df: pd.DataFrame, filename: str, title: str = "Tabela"):
    """
    Salva um DataFrame como arquivo Excel e CSV no diretório temporário.
    """
    if df is None or df.empty:
        return

    export_dir = _init_exports(scope_key)
    base_path = os.path.join(export_dir, filename)

    # salva como Excel
    excel_path = f"{base_path}.xlsx"
    try:
        df.to_excel(excel_path, index=False, engine="openpyxl")
    except Exception:
        df.to_csv(f"{base_path}.csv", index=False, encoding="utf-8-sig")

    st.success(f"✅ {title} exportada ({filename})")


# ---------------------------------------------------------------
# 🖼️ Exporta e exibe figuras do matplotlib
# ---------------------------------------------------------------
def show_and_export_fig(scope_key: str, fig: plt.Figure, filename: str):
    """
    Mostra o gráfico no Streamlit e salva PNG/HTML para download posterior.
    """
    export_dir = _init_exports(scope_key)
    png_path = os.path.join(export_dir, f"{filename}.png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    st.pyplot(fig, use_container_width=True)
    st.caption(f"📁 Figura salva: {filename}.png")


# ---------------------------------------------------------------
# 📦 Gera botão de download .zip
# ---------------------------------------------------------------
def export_zip_button(scope_key: str):
    """
    Agrupa todos os arquivos do diretório temporário no escopo atual
    e gera um botão de download .zip.
    """
    export_dir = _init_exports(scope_key)
    zip_buffer = io.BytesIO()
    zip_path = os.path.join(export_dir, f"export_{scope_key}.zip")

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(export_dir):
            for f in files:
                file_path = os.path.join(root, f)
                zipf.write(file_path, arcname=f)

    zip_buffer.seek(0)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    st.download_button(
        label=f"⬇️ Baixar resultados ({now})",
        data=zip_buffer,
        file_name=f"EmentaLabv2_{scope_key}_{now}.zip",
        mime="application/zip"
    )


# ---------------------------------------------------------------
# 🧹 Limpeza manual do diretório (opcional)
# ---------------------------------------------------------------
def clear_exports(scope_key: str = "default"):
    """
    Limpa os arquivos temporários de exportação para o escopo informado.
    """
    export_dir = _init_exports(scope_key)
    try:
        for f in os.listdir(export_dir):
            os.remove(os.path.join(export_dir, f))
        st.info(f"🧹 Arquivos temporários limpos ({scope_key}).")
    except Exception as e:
        st.warning(f"Não foi possível limpar arquivos: {e}")


# ---------------------------------------------------------------
# 🧮 Utilitário auxiliar para múltiplas figuras
# ---------------------------------------------------------------
def export_multiple_figs(scope_key: str, figs: dict):
    """
    Exporta múltiplas figuras (dict nome:figura) para PNGs.
    """
    export_dir = _init_exports(scope_key)
    for name, fig in figs.items():
        path = os.path.join(export_dir, f"{name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    st.success(f"📦 {len(figs)} figuras exportadas em {export_dir}")
