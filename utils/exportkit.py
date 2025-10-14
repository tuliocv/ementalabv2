# ===============================================================
# 💾 EmentaLabv2 — ExportKit Utilitário (v11.2)
# ===============================================================
# Responsável por inicializar diretórios temporários de exportação,
# salvar tabelas, gráficos e gerar pacotes .zip de resultados.
# ---------------------------------------------------------------
# ✅ Compatível com Streamlit Cloud
# ✅ Evita IDs duplicados de download_button
# ✅ Mantém compatibilidade com versões anteriores
# ===============================================================

import os
import io
import uuid
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

    # 🔹 limpeza automática de arquivos antigos (> 12h)
    _cleanup_old_exports(tempfile.gettempdir())

    return export_dir


# ---------------------------------------------------------------
# 🧹 Limpeza automática de diretórios temporários antigos
# ---------------------------------------------------------------
def _cleanup_old_exports(base_tmp):
    """Remove pastas antigas do EmentaLab com mais de 12h de criação."""
    import time
    now = time.time()
    for f in os.listdir(base_tmp):
        if f.startswith("ementalab_exports_"):
            path = os.path.join(base_tmp, f)
            try:
                if now - os.path.getmtime(path) > 12 * 3600:
                    for ff in os.listdir(path):
                        os.remove(os.path.join(path, ff))
                    os.rmdir(path)
            except Exception:
                pass


# ---------------------------------------------------------------
# 💾 Exporta DataFrame como Excel/CSV
# ---------------------------------------------------------------
def export_table(scope_key: str, df: pd.DataFrame, filename: str, title: str = "Tabela"):
    """
    Salva um DataFrame como arquivo Excel e CSV no diretório temporário.
    """
    if df is None or df.empty:
        st.warning(f"⚠️ Nenhum dado disponível para exportar ({title}).")
        return

    export_dir = _init_exports(scope_key)
    base_path = os.path.join(export_dir, filename)

    try:
        df.to_excel(f"{base_path}.xlsx", index=False, engine="openpyxl")
    except Exception:
        df.to_csv(f"{base_path}.csv", index=False, encoding="utf-8-sig")

    st.success(f"✅ {title} exportada com sucesso ({filename})")


# ---------------------------------------------------------------
# 🖼️ Exporta e exibe figuras do matplotlib
# ---------------------------------------------------------------
def show_and_export_fig(scope_key: str, fig: plt.Figure, filename: str, show=True):
    """
    Mostra o gráfico no Streamlit e salva PNG para download posterior.
    """
    export_dir = _init_exports(scope_key)
    png_path = os.path.join(export_dir, f"{filename}.png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)

    if show:
        st.pyplot(fig, use_container_width=True)
        st.caption(f"📁 Figura salva: `{filename}.png`")


# ---------------------------------------------------------------
# 📦 Gera botão de download .zip (com chave única)
# ---------------------------------------------------------------
def export_zip_button(scope_key: str):
    """
    Agrupa todos os arquivos do diretório temporário no escopo atual
    e gera um botão de download .zip com identificador único.
    """
    export_dir = _init_exports(scope_key)
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(export_dir):
            for f in files:
                file_path = os.path.join(root, f)
                zipf.write(file_path, arcname=f)

    zip_buffer.seek(0)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 🔹 gera chave única para evitar IDs duplicados no Streamlit
    unique_key = f"download_{scope_key}_{uuid.uuid4().hex[:8]}"

    st.download_button(
        label=f"⬇️ Baixar resultados ({now})",
        data=zip_buffer,
        file_name=f"EmentaLabv2_{scope_key}_{now}.zip",
        mime="application/zip",
        key=unique_key,
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
