# ===============================================================
# 💾 EmentaLabv2 — ExportKit Utilitário (v11.5)
# ===============================================================
# Solução final contra StreamlitDuplicateElementKey
# ---------------------------------------------------------------
# ✅ "Baixar Resultados" único por escopo, sem duplicar na tela
# ✅ Identificador interno aleatório (garante unicidade)
# ✅ Compatível com múltiplos módulos e Streamlit Cloud
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
    """Cria (ou reutiliza) diretório temporário de exportação."""
    export_dir = os.path.join(tempfile.gettempdir(), f"ementalab_exports_{scope_key}")
    os.makedirs(export_dir, exist_ok=True)
    _cleanup_old_exports(tempfile.gettempdir())
    return export_dir


# ---------------------------------------------------------------
# 🧹 Limpa diretórios antigos
# ---------------------------------------------------------------
def _cleanup_old_exports(base_tmp):
    """Remove pastas antigas com mais de 12h."""
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
    """Salva DataFrame como Excel ou CSV."""
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
    """Mostra o gráfico no Streamlit e salva PNG."""
    export_dir = _init_exports(scope_key)
    png_path = os.path.join(export_dir, f"{filename}.png")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    if show:
        st.pyplot(fig, use_container_width=True)
        st.caption(f"📁 Figura salva: `{filename}.png`")


# ---------------------------------------------------------------
# 📦 Gera botão de download .zip (único visualmente)
# ---------------------------------------------------------------
def export_zip_button(scope_key: str):
    """
    Gera um botão fixo "Baixar Resultados".
    Garante unicidade mesmo com múltiplas chamadas simultâneas.
    """
    export_dir = _init_exports(scope_key)

    # 🔸 verifica se botão já foi exibido neste render
    if "_shown_buttons" not in st.session_state:
        st.session_state["_shown_buttons"] = set()
    if scope_key in st.session_state["_shown_buttons"]:
        return
    st.session_state["_shown_buttons"].add(scope_key)

    # 🔸 gera conteúdo ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(export_dir):
            for f in files:
                zipf.write(os.path.join(root, f), arcname=f)
    zip_buffer.seek(0)

    # 🔸 gera key única, sem repetição
    unique_key = f"dl_{scope_key}_{uuid.uuid4().hex[:6]}"

    # 🔸 exibe botão fixo (único por escopo)
    st.download_button(
        label="⬇️ Baixar Resultados",
        data=zip_buffer,
        file_name=f"EmentaLabv2_{scope_key}.zip",
        mime="application/zip",
        key=unique_key,
    )


# ---------------------------------------------------------------
# 🧹 Limpa diretório manualmente
# ---------------------------------------------------------------
def clear_exports(scope_key: str = "default"):
    """Remove todos os arquivos exportados manualmente."""
    export_dir = _init_exports(scope_key)
    try:
        for f in os.listdir(export_dir):
            os.remove(os.path.join(export_dir, f))
        st.info(f"🧹 Arquivos temporários limpos ({scope_key}).")
    except Exception as e:
        st.warning(f"Não foi possível limpar arquivos: {e}")


# ---------------------------------------------------------------
# 🧮 Exporta múltiplas figuras
# ---------------------------------------------------------------
def export_multiple_figs(scope_key: str, figs: dict):
    """Exporta várias figuras para o diretório temporário."""
    export_dir = _init_exports(scope_key)
    for name, fig in figs.items():
        path = os.path.join(export_dir, f"{name}.png")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    st.success(f"📦 {len(figs)} figuras exportadas em {export_dir}")
