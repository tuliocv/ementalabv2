# ===============================================================
# 📈 EmentaLabv2 — Clusterização (Ementa)
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, show_and_export_fig, export_zip_button
from utils.text_utils import find_col, replace_semicolons

# ---------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------
def run_cluster(df, scope_key):
    st.markdown("<h2 style='color:#2ca02c;'>📈 Clusterização de Ementas</h2>", unsafe_allow_html=True)
    st.caption(
        """
        Esta análise agrupa as **Ementas** de Unidades Curriculares (UCs) com base em similaridade semântica,
        utilizando embeddings SBERT e o algoritmo K-Means.  
        O objetivo é identificar **núcleos temáticos** e **possíveis redundâncias ou convergências curriculares**.
        """
    )

    # -----------------------------------------------------------
    # 📂 Seleção da coluna base
    # -----------------------------------------------------------
    col_ementa = find_col(df, "Ementa")
    if not col_ementa:
        st.error("Coluna 'Ementa' não encontrada.")
        st.stop()

    df_an = df.dropna(subset=[col_ementa])
    textos = df_an[col_ementa].astype(str).apply(replace_semicolons).tolist()
    nomes = df_an["Nome da UC"].astype(str).tolist()

    # -----------------------------------------------------------
    # 🧠 Geração de embeddings (com cache)
    # -----------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def get_embeddings(textos):
        return l2_normalize(sbert_embed(textos))

    with st.spinner("🧠 Calculando embeddings semânticos (SBERT)..."):
        emb = get_embeddings(textos)

    # -----------------------------------------------------------
    # 📉 Sugestão automática de número de clusters (Elbow Method)
    # -----------------------------------------------------------
    st.markdown("### 🔢 Definição do número de clusters (K)")

    if st.checkbox("Sugerir K automaticamente (método do cotovelo)", value=False):
        distortions = []
        K_range = range(2, 10)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(emb)
            distortions.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K_range, distortions, marker="o", color="#2ca02c")
        ax.set_xlabel("Número de Clusters (K)")
        ax.set_ylabel("Soma dos Erros Quadráticos (Inércia)")
        ax.set_title("Método do Cotovelo (Elbow Method)")
        show_and_export_fig(scope_key, fig, "cluster_elbow_method")
        st.info("📊 Observe onde ocorre a curva (cotovelo). Esse ponto indica o número ideal de clusters.")

    # -----------------------------------------------------------
    # 🧩 Definição manual de K
    # -----------------------------------------------------------
    k = st.slider("Número de clusters (K)", 2, 10, 5)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(emb)

    df_out = pd.DataFrame({"Nome da UC": nomes, "Cluster": labels})
    df_out = df_out.merge(df_an[[col_ementa, "Nome da UC"]], on="Nome da UC", how="left")

    # -----------------------------------------------------------
    # 📊 Visualização PCA 2D
    # -----------------------------------------------------------
    st.markdown("### 🧭 Visualização dos Clusters (PCA 2D)")
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(emb)
    df_pca = pd.DataFrame({"x": reduced[:, 0], "y": reduced[:, 1], "Cluster": labels, "Nome da UC": nomes})

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="x", y="y", hue="Cluster", palette="tab10", s=80, ax=ax)
    ax.set_title("Projeção dos Clusters de Ementas (PCA)")
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.legend(title="Cluster")
    show_and_export_fig(scope_key, fig, "cluster_pca_2d")

    # -----------------------------------------------------------
    # 🧩 Resumo de palavras-chave por cluster
    # -----------------------------------------------------------
    st.markdown("### 🧩 Tópicos predominantes por Cluster")

    cluster_keywords = []
    vectorizer = CountVectorizer(stop_words="portuguese", max_features=1000)
    X = vectorizer.fit_transform(textos)
    words = np.array(vectorizer.get_feature_names_out())

    for c in range(k):
        mask = (labels == c)
        cluster_texts = X[mask].sum(axis=0).A1
        top_idx = cluster_texts.argsort()[-10:][::-1]
        top_words = words[top_idx]
        cluster_keywords.append({"Cluster": c, "Palavras-Chave": ", ".join(top_words)})

    df_keywords = pd.DataFrame(cluster_keywords)
    st.dataframe(df_keywords, use_container_width=True)

    # -----------------------------------------------------------
    # 💾 Exportação dos resultados
    # -----------------------------------------------------------
    export_table(scope_key, df_out, "clusterizacao", "Clusters (Ementa)")
    export_table(scope_key, df_keywords, "cluster_keywords", "Palavras-Chave por Cluster")
    export_zip_button(scope_key)

    # -----------------------------------------------------------
    # 📘 Interpretação pedagógica
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("📘 Como interpretar os resultados")
    st.markdown(
        """
        **1️⃣ O que representa cada cluster:**
        - Cada grupo reúne UCs com **ementas semanticamente semelhantes**.
        - UCs no mesmo cluster tendem a compartilhar **conteúdos, competências e abordagens formativas**.

        **2️⃣ Como usar os resultados:**
        - Identifique **sobreposição temática** entre UCs do mesmo cluster (possível redundância curricular).
        - Analise se clusters distintos correspondem a **núcleos de formação** coerentes (ex.: “Gestão”, “Cálculo”, “Programação”).
        - Compare os clusters com a estrutura do curso (Matriz Curricular) para **alinhar competências e conteúdos.**

        **3️⃣ Aplicações práticas:**
        - Revisar ementas redundantes ou excessivamente próximas.
        - Mapear a **convergência interdisciplinar** entre cursos ou unidades curriculares.
        - Priorizar integração entre clusters relacionados (núcleos integradores).
        """
    )
