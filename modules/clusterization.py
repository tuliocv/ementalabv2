# ===============================================================
# 📈 EmentaLabv2 — Clusterização (Ementas) + Nomeação via GPT
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from openai import OpenAI

from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, show_and_export_fig, export_zip_button
from utils.text_utils import find_col, replace_semicolons

# ---------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------
def run_cluster(df, scope_key):
    # -----------------------------------------------------------
    # 🏷️ Título e descrição
    # -----------------------------------------------------------
    st.markdown(
        "<h2 style='color:#2ca02c;'>📈 Clusterização de Ementas</h2>",
        unsafe_allow_html=True
    )
    st.caption(
        """
        Esta análise agrupa as **Ementas das Unidades Curriculares (UCs)** com base em similaridade semântica.
        Utiliza **embeddings SBERT** e o algoritmo **K-Means** para revelar **núcleos temáticos** e **áreas de convergência curricular**.
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
    # 🔢 Definição do número de clusters
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
        st.info("📊 Observe o ponto de inflexão do gráfico (cotovelo). Ele indica o K mais adequado.")

    # -----------------------------------------------------------
    # 🧩 Clusterização manual
    # -----------------------------------------------------------
    k = st.slider("Número de clusters (K)", 2, 10, 5)
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(emb)

    df_out = pd.DataFrame({
        "Nome da UC": nomes,
        "Cluster": labels,
        "Ementa": textos
    })

    # -----------------------------------------------------------
    # 🧮 Determinação da UC representativa (mais próxima do centróide)
    # -----------------------------------------------------------
    representative_ucs = []
    for c in range(k):
        cluster_points = emb[labels == c]
        centroid = km.cluster_centers_[c]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        idx = np.argmin(distances)
        uc_name = df_out[df_out["Cluster"] == c]["Nome da UC"].iloc[idx]
        representative_ucs.append(uc_name)

    # -----------------------------------------------------------
    # 🧩 Palavras-chave por cluster (corrigido)
    # -----------------------------------------------------------
    st.markdown("### 🧩 Tópicos predominantes por Cluster")

    # ✅ Lista de stopwords compatível com qualquer versão do sklearn
    base_stopwords = list(ENGLISH_STOP_WORDS)
    extra_stopwords_pt = [
        "de", "da", "do", "das", "dos", "para", "por", "com", "a", "o", "e", "em",
        "como", "ao", "na", "no", "nas", "nos", "sobre", "entre", "pelas", "pelos",
        "pelo", "pela", "ser", "estar", "ter", "se", "que", "onde", "quando", "uma",
        "um", "as", "os", "é", "das", "dos", "nas", "nos"
    ]
    all_stopwords = base_stopwords + extra_stopwords_pt

    vectorizer = CountVectorizer(
        stop_words=all_stopwords,
        max_features=1000,
        token_pattern=r"(?u)\b\w\w+\b"
    )

    X = vectorizer.fit_transform(textos)
    words = np.array(vectorizer.get_feature_names_out())

    cluster_keywords = []
    for c in range(k):
        mask = (labels == c)
        cluster_texts = X[mask].sum(axis=0).A1
        top_idx = cluster_texts.argsort()[-10:][::-1]
        top_words = words[top_idx]
        cluster_keywords.append({
            "Cluster": c,
            "UC Representativa": representative_ucs[c],
            "Palavras-Chave": ", ".join(top_words)
        })

    df_keywords = pd.DataFrame(cluster_keywords)
    st.dataframe(df_keywords, use_container_width=True)

    # -----------------------------------------------------------
    # 🧭 Visualização PCA 2D
    # -----------------------------------------------------------
    st.markdown("### 🧭 Visualização dos Clusters (PCA 2D)")
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(emb)
    df_pca = pd.DataFrame({
        "x": reduced[:, 0], "y": reduced[:, 1],
        "Cluster": labels, "Nome da UC": nomes
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="x", y="y", hue="Cluster", palette="tab10", s=80, ax=ax)
    ax.set_title("Projeção dos Clusters de Ementas (PCA)")
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.legend(title="Cluster")
    show_and_export_fig(scope_key, fig, "cluster_pca_2d")

    # -----------------------------------------------------------
    # 🤖 Nomeação automática dos clusters (GPT opcional)
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Nomeação Inteligente dos Clusters (opcional)")

    api_key = st.text_input("🔑 OpenAI API Key (opcional para nomeação GPT)", type="password")
    if api_key:
        client = OpenAI(api_key=api_key)
        st.info("O modelo GPT analisará os tópicos e sugerirá nomes representativos para cada cluster.")
        cluster_names = []

        with st.spinner("🧠 Gerando nomes sugestivos para os clusters..."):
            for _, row in df_keywords.iterrows():
                prompt = f"""
                Você é um especialista em educação superior e análise curricular.
                Dê um nome temático curto e representativo para o seguinte cluster de disciplinas:

                UC representativa: {row['UC Representativa']}
                Palavras-chave: {row['Palavras-Chave']}

                Responda apenas com o nome proposto (sem explicações adicionais).
                """
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    nome_cluster = resp.choices[0].message.content.strip().replace('"', '')
                    cluster_names.append(nome_cluster)
                except Exception as e:
                    cluster_names.append(f"Erro: {str(e)}")

        df_keywords["Nome GPT"] = cluster_names
        st.dataframe(df_keywords, use_container_width=True)
        export_table(scope_key, df_keywords, "cluster_keywords_gpt", "Clusters Nomeados GPT")
    else:
        st.info("Insira sua chave de API para permitir que o GPT nomeie os clusters.")

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
        **1️⃣ Significado dos clusters:**
        - Cada grupo reúne UCs com **ementas semanticamente semelhantes**.
        - UCs próximas compartilham **conteúdos, abordagens e competências** similares.

        **2️⃣ Interpretação prática:**
        - Clusters grandes indicam **núcleos formativos amplos** (ex.: Matemática, Programação, Gestão).  
        - Clusters pequenos podem sinalizar **especializações** ou **redundâncias curriculares**.  
        - A UC representativa indica **a disciplina mais central** dentro do tema.

        **3️⃣ Uso com GPT:**
        - O nome sugerido pelo GPT ajuda a **etiquetar os núcleos temáticos** de forma interpretável.
        - Ideal para relatórios de análise curricular, consolidação de PPCs e reuniões de NDE.

        **4️⃣ Aplicações práticas:**
        - Diagnóstico de **redundância e sobreposição curricular**.
        - Identificação de **áreas interdisciplinares** emergentes.
        - Planejamento de **integração entre clusters correlatos**.
        """
    )
