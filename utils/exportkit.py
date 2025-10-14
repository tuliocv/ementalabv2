# ===============================================================
# 💾 EmentaLabv2 — ExportKit Utilitário (v11.6)
# ===============================================================
# 🔹 Solução final contra StreamlitDuplicateElementKey
# 🔹 Adicionada compatibilidade retroativa com get_docx_bytes()
# 🔹 Log opcional de exportações (arquivos incluídos no ZIP)
# ---------------------------------------------------------------
# ✅ "Baixar Resultados" único por escopo
# ✅ Identificador interno aleatório (garante unicidade)
# ✅ Compatível com múltiplos módulos e Streamlit Cloud
# ✅ Inclui suporte a .docx (para módulos legados)
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
# 🧹 Limpa diretórios antigos (mais de 12h)
# ---------------------------------------------------------------
def _cleanup_old_exports(base_tmp):
    """Remove pastas antigas de exportação."""
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
    """Salva DataFrame como Excel ou CSV no diretório de exportação."""
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
    """Mostra o gráfico e salva como PNG."""
    export_dir = _init_exports(scope_key)
    png_path = os.path.join(export_dir, f"{filename}.png")

    try:
        fig.savefig(png_path, bbox_inches="tight", dpi=300)
        if show:
            st.pyplot(fig, use_container_width=True)
            st.caption(f"📁 Figura salva: `{filename}.png`")
    except Exception as e:
        st.error(f"❌ Erro ao exportar figura: {e}")


# ---------------------------------------------------------------
# 📦 Gera botão de download .zip (único por escopo)
# ---------------------------------------------------------------
def export_zip_button(scope_key: str):
    """
    Gera um botão fixo "Baixar Resultados".
    Evita duplicações e cria chave única a cada render.
    """
    export_dir = _init_exports(scope_key)

    # 🔸 Evita botões duplicados
    if "_shown_buttons" not in st.session_state:
        st.session_state["_shown_buttons"] = set()
    if scope_key in st.session_state["_shown_buttons"]:
        return
    st.session_state["_shown_buttons"].add(scope_key)

    # 🔸 Gera o ZIP
    zip_buffer = io.BytesIO()
    added_files = []
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(export_dir):
            for f in files:
                full_path = os.path.join(root, f)
                zipf.write(full_path, arcname=f)
                added_files.append(f)
    zip_buffer.seek(0)

    # 🔸 Log opcional no console
    print(f"[EmentaLabv2][ExportKit] Arquivos exportados ({scope_key}): {', '.join(added_files)}")

    # 🔸 Cria botão único com key randômica
    unique_key = f"dl_{scope_key}_{uuid.uuid4().hex[:6]}"
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
    """Exporta várias figuras simultaneamente."""
    export_dir = _init_exports(scope_key)
    count = 0
    for name, fig in figs.items():
        try:
            path = os.path.join(export_dir, f"{name}.png")
            fig.savefig(path, bbox_inches="tight", dpi=300)
            count += 1
        except Exception as e:
            st.error(f"❌ Erro ao salvar figura '{name}': {e}")
    st.success(f"📦 {count} figuras exportadas para {export_dir}")


# ---------------------------------------------------------------
# 🧾 Compatibilidade retroativa — exportação DOCX
# ---------------------------------------------------------------
from io import BytesIO

def get_docx_bytes(document):
    """
    Compatibilidade com módulos antigos (EmentaLabv1).
    Converte um objeto `docx.Document` em bytes prontos para download.
    Exemplo:
        buffer = get_docx_bytes(doc)
        st.download_button("Baixar DOCX", data=buffer, file_name="relatorio.docx")
    """
    bio = BytesIO()
    try:
        document.save(bio)
        bio.seek(0)
        return bio.getvalue()
    except Exception as e:
        st.error(f"❌ Erro ao gerar arquivo DOCX: {e}")
        return None
