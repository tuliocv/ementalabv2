# ============================================================
# ☁️ WordCloud Analysis — Nuvem de Palavras das UCs
# ============================================================
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re

# ============================================================
# 🚀 Função Principal
# ============================================================
def run_wordcloud(df: pd.DataFrame, scope_key: str):
    """
    Gera uma nuvem de palavras com base nas ementas, objetivos ou competências das UCs.
    """

    st.subheader("☁️ Nuvem de Palavras das Unidades Curriculares")

    # --------------------------------------------------------
    # 🧭 Seleção do campo base
    # --------------------------------------------------------
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    col_sel = st.selectbox(
        "Selecione o campo de texto para gerar a nuvem:",
        options=text_cols,
        index=text_cols.index("Ementa") if "Ementa" in text_cols else 0
    )

    # --------------------------------------------------------
    # 🧹 Pré-processamento textual
    # --------------------------------------------------------
    def limpar_texto(txt):
        txt = str(txt).lower()
        txt = re.sub(r"http\S+|www\S+|https\S+", "", txt)
        txt = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", txt)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()

    df["__texto_limpo__"] = df[col_sel].astype(str).apply(limpar_texto)
    texto_total = " ".join(df["__texto_limpo__"].tolist())

    # --------------------------------------------------------
    # ⚙️ Configurações de exibição
    # --------------------------------------------------------
    st.sidebar.markdown("### 🎨 Configurações da Nuvem")
    max_palavras = st.sidebar.slider("Máximo de palavras", 50, 500, 200, 50)
    colormap = st.sidebar.selectbox("Esquema de cores", [
        "viridis", "plasma", "inferno", "magma", "cividis",
        "Blues", "Greens", "Oranges", "Purples", "Reds",
        "cool", "hot", "rainbow"
    ], index=0)
    fundo_transp = st.sidebar.checkbox("Fundo transparente", value=False)

    # --------------------------------------------------------
    # ☁️ Geração da nuvem
    # --------------------------------------------------------
    if not texto_total.strip():
        st.warning("Nenhum texto disponível para gerar a nuvem.")
        return

    st.info(f"Gerando nuvem de palavras com base em **{col_sel}**...")

    wc = WordCloud(
        width=1200,
        height=600,
        background_color=None if fundo_transp else "white",
        mode="RGBA" if fundo_transp else "RGB",
        colormap=colormap,
        max_words=max_palavras,
        collocations=False,
        normalize_plurals=True
    ).generate(texto_total)

    # --------------------------------------------------------
    # 📊 Exibir gráfico
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # --------------------------------------------------------
    # 📦 Download opcional
    # --------------------------------------------------------
    buf = wc.to_image()
    st.download_button(
        "💾 Baixar imagem da nuvem",
        data=_image_to_bytes(buf),
        file_name=f"nuvem_palavras_{scope_key}.png",
        mime="image/png"
    )

# ============================================================
# 🔧 Função auxiliar para converter imagem em bytes
# ============================================================
from io import BytesIO
def _image_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
