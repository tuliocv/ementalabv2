# ============================================================
# ☁️ WordCloud Analysis — Nuvem de Palavras das UCs
# ============================================================
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import re

# ============================================================
# 🚀 Função Principal
# ============================================================
def run_wordcloud(df: pd.DataFrame, scope_key: str):
    """
    Gera nuvens de palavras específicas para colunas curriculares
    (Competências, Ementa, Objetivos etc.), com remoção de stopwords.
    """

    st.subheader("☁️ Nuvem de Palavras das Unidades Curriculares")

    # --------------------------------------------------------
    # 🧭 Colunas de interesse específicas
    # --------------------------------------------------------
    colunas_alvo = [
        "Competências do Perfil do Egresso",
        "Relação competência DCN",
        "Ementa (Assuntos que serão abordado)",
        "Objetivo de aprendizagem",
        "Objetos de conhecimento (Conteúdo programático)",
    ]

    # Filtra colunas que existem no DataFrame
    colunas_existentes = [c for c in colunas_alvo if c in df.columns]

    if not colunas_existentes:
        st.warning("Nenhuma das colunas esperadas foi encontrada na base.")
        st.write("Esperadas:", ", ".join(colunas_alvo))
        return

    # --------------------------------------------------------
    # 🧹 Pré-processamento textual
    # --------------------------------------------------------
    def limpar_texto(txt):
        txt = str(txt).lower()
        txt = re.sub(r"http\S+|www\S+|https\S+", "", txt)
        txt = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", txt)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()

    # --------------------------------------------------------
    # 🧾 Stopwords em português
    # --------------------------------------------------------
    stopwords_pt = set([
        # stopwords básicas
        "de", "da", "do", "das", "dos", "e", "em", "para", "por", "com", "a", "o", "as", "os", 
        "na", "no", "nas", "nos", "um", "uma", "uns", "umas", "que", "como", "se", "ao", "à",
        "às", "aos", "sobre", "entre", "pela", "pelo", "pelas", "pelos", "não", "ser", "estar",
        "ter", "ou", "sua", "seu", "suas", "seus", "mais", "menos", "também", "quando", "onde",
        "cada", "outro", "outra", "outros", "outras", "para", "com", "pode", "podem", "através",
        "visando", "formação", "capacidade", "habilidade", "competência", "conhecimento"
    ]) | STOPWORDS

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
    # ☁️ Geração das nuvens para cada coluna
    # --------------------------------------------------------
    for col in colunas_existentes:
        st.markdown(f"### 📘 {col}")
        df["__texto_limpo__"] = df[col].astype(str).apply(limpar_texto)
        texto_total = " ".join(df["__texto_limpo__"].tolist())

        if not texto_total.strip():
            st.info(f"Nenhum texto válido encontrado na coluna **{col}**.")
            continue

        wc = WordCloud(
            width=1200,
            height=600,
            background_color=None if fundo_transp else "white",
            mode="RGBA" if fundo_transp else "RGB",
            colormap=colormap,
            max_words=max_palavras,
            stopwords=stopwords_pt,
            collocations=False,
            normalize_plurals=True
        ).generate(texto_total)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # 📦 Download individual
        buf = wc.to_image()
        st.download_button(
            f"💾 Baixar imagem — {col}",
            data=_image_to_bytes(buf),
            file_name=f"nuvem_{col.replace(' ', '_')}_{scope_key}.png",
            mime="image/png"
        )
        st.markdown("---")

# ============================================================
# 🔧 Função auxiliar para converter imagem em bytes
# ============================================================
from io import BytesIO
def _image_to_bytes(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
