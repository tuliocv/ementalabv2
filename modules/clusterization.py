# ===============================================================
# 📈 EmentaLabv2 — Clusterização (Ementa) c/ UC_ID, Nome GPT e Comparação
# ===============================================================
import json
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

# Utils do projeto
from utils.embeddings import l2_normalize, sbert_embed
from utils.exportkit import export_table, show_and_export_fig, export_zip_button
from utils.text_utils import find_col, replace_semicolons

# -------------- Helpers -----------------

def _clean_corpus(textos: list[str]) -> list[str]:
    """Limpa e padroniza os textos."""
    out = []
    for t in textos:
        s = str(t)
        s = re.sub(r"[\r\n\t]+", " ", s)      # quebras
        s = re.sub(r"\s{2,}", " ", s).strip() # espaços
        out.append(s)
    return out

def _tfidf_top_keywords_per_cluster(textos: list[str], labels: np.ndarray, top_k: int = 8) -> pd.DataFrame:
    """
    Retorna um DataFrame com palavras-chave por cluster.
    Usa parâmetros compatíveis com sklearn 1.4+ (evita InvalidParameterError).
    """
    if len(textos) == 0:
        return pd.DataFrame(columns=["Cluster", "Palavras-chave"])

    textos_clean = _clean_corpus(textos)

    # Vectorizer enxuto e compatível
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=8000,
        norm="l2"
    )

    X = vectorizer.fit_transform(textos_clean)
    vocab = np.array(vectorizer.get_feature_names_out())

    df_kw = []
    k = int(labels.max()) + 1 if len(labels) else 0
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            df_kw.append({"Cluster": cid, "Palavras-chave": []})
            continue
        sub = X[idx].mean(axis=0)  # média TF-IDF do cluster
        sub = np.asarray(sub).ravel()
        if sub.sum() == 0:
            df_kw.append({"Cluster": cid, "Palavras-chave": []})
            continue
        top_idx = np.argsort(-sub)[:top_k]
        kws = [vocab[i] for i in top_idx]
        df_kw.append({"Cluster": cid, "Palavras-chave": kws})

    return pd.DataFrame(df_kw)

def _representative_uc_by_centroid(embeddings: np.ndarray, labels: np.ndarray, nomes: list[str]) -> dict:
    """
    Para cada cluster, encontra a UC cujo embedding está mais próximo do centróide (menor distância).
    Retorna {cluster_id: {"uc": Nome, "idx": índice}}.
    """
    reps = {}
    k = int(labels.max()) + 1 if len(labels) else 0
    for cid in range(k):
        idxs = np.where(labels == cid)[0]
        if idxs.size == 0:
            reps[cid] = {"uc": f"Cluster {cid}", "idx": None}
            continue
        C = embeddings[idxs].mean(axis=0, keepdims=True)
        d = cosine_distances(embeddings[idxs], C).ravel()
        best_local = idxs[np.argmin(d)]
        reps[cid] = {"uc": nomes[best_local], "idx": int(best_local)}
    return reps

def _project_2d(emb: np.ndarray) -> np.ndarray:
    """Tenta UMAP → T-SNE → SVD, nessa ordem."""
    try:
        import umap
        return umap.UMAP(n_neighbors=12, min_dist=0.1, random_state=42).fit_transform(emb)
    except Exception:
        try:
            from sklearn.manifold import TSNE
            perplex = max(2, min(30, emb.shape[0] // 3 if emb.shape[0] >= 6 else 2))
            return TSNE(n_components=2, perplexity=perplex, random_state=42, init="random", learning_rate="auto").fit_transform(emb)
        except Exception:
            from sklearn.decomposition import TruncatedSVD
            return TruncatedSVD(n_components=2, random_state=42).fit_transform(emb)

def _plot_scatter(df_plot: pd.DataFrame, title: str, label_col: str, scope_key: str, base_name: str):
    """Desenha scatter 2D com rótulos UC_ID e legenda por rótulos de cluster."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    labels = df_plot[label_col].astype(str).unique().tolist()
    palette = sns.color_palette("husl", len(labels))

    for i, lab in enumerate(labels):
        sub = df_plot[df_plot[label_col].astype(str) == str(lab)]
        ax.scatter(sub["X"], sub["Y"], s=70, alpha=0.9, color=palette[i], label=str(lab), edgecolor="white", linewidths=0.6)
        # anota UC_ID no ponto
        for _, row in sub.iterrows():
            ax.text(row["X"], row["Y"], str(int(row["UC_ID"])), fontsize=8, ha="center", va="center", color="black", alpha=0.9, clip_on=False)

    ax.set_title(title)
    ax.set_xlabel("Dimensão 1")
    ax.set_ylabel("Dimensão 2")
    ax.legend(title=label_col, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.margins(0.05)

    show_and_export_fig(scope_key, fig, base_name)
    plt.close(fig)

def _call_gpt_for_names(clusters_payload: list[dict], api_key: str) -> dict:
    """
    Envia um resumo de cada cluster para o GPT sugerir nomes.
    Retorna dict {cluster_id: "Nome GPT"}.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        st.error("❌ Não foi possível inicializar o cliente OpenAI. Verifique se o pacote `openai` está instalado e a chave é válida.")
        return {}

    prompt = (
        "Você receberá um conjunto de clusters de ementas. Para cada cluster, atribua um **nome temático curto e claro** "
        "(máx. 4 palavras), em português, que represente o conteúdo central.\n"
        "Responda **apenas** com JSON no formato:\n"
        "{ \"nomes\": [ {\"cluster\": <id>, \"nome\": \"<Nome do cluster>\"}, ... ] }\n\n"
        "Clusters:\n"
    )
    for c in clusters_payload:
        prompt += f"- Cluster {c['cluster']}: Representante = {c['representante']}; Keywords = {', '.join(c['keywords'][:8])}; Exemplo de ementa: {c['amostra_ementa']}\n"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        nomes = {}
        for item in data.get("nomes", []):
            cid = int(item.get("cluster"))
            nome = str(item.get("nome", "")).strip()
            if nome:
                nomes[cid] = nome
        return nomes
    except Exception as e:
        st.error(f"❌ Erro ao consultar GPT para nomear clusters: {e}")
        return {}

# -------------- Módulo principal -----------------

def run_cluster(df: pd.DataFrame, scope_key: str):
    """
    Clusterização de ementas com:
      - UC_ID sequencial
      - KMeans (SBERT)
      - Palavras-chave TF-IDF por cluster
      - Nome do cluster = UC representante (centróide)
      - Nome GPT opcional + gráfico comparativo
      - Tabela consolidada com todas as UCs
    """
    st.header("📈 Clusterização (Ementa)")
    st.markdown("""
**O que é:** agrupa UCs por similaridade semântica das **Ementas** e projeta em 2D (cada ponto = 1 UC).  
**Como analisar:** pontos próximos compartilham tema; use os **números (UC_ID)** para localizar a UC na tabela.  
**Dicas:** ajuste **K**, confira as **palavras-chave** por cluster e, se quiser, peça ao **GPT** nomes temáticos.
    """)

    # --------- Coluna base ---------
    col_ementa = find_col(df, "Ementa")
    if not col_ementa:
        st.error("Coluna 'Ementa' não encontrada. Verifique o cabeçalho.")
        st.stop()

    df_an = df.dropna(subset=[col_ementa]).copy()
    if df_an.empty:
        st.warning("Não há ementas válidas para clusterizar.")
        st.stop()

    # Numeração sequencial
    df_an = df_an.reset_index(drop=True)
    df_an["UC_ID"] = np.arange(1, len(df_an) + 1)

    textos = df_an[col_ementa].astype(str).apply(replace_semicolons).tolist()
    nomes = df_an["Nome da UC"].astype(str).tolist()

    # --------- Embeddings ---------
    with st.spinner("🧠 Calculando embeddings SBERT..."):
        emb = l2_normalize(sbert_embed(textos).astype(np.float32))

    # --------- KMeans ---------
    max_k = min(12, max(2, len(df_an)))
    k = st.slider("Número de clusters (K)", 2, max_k, min(6, max_k))
    with st.spinner("🧮 Executando K-Means..."):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(emb)

    # --------- Representantes e keywords ---------
    with st.spinner("🔎 Gerando representantes e palavras-chave..."):
        reps = _representative_uc_by_centroid(emb, labels, nomes)
        df_kw = _tfidf_top_keywords_per_cluster(textos, labels, top_k=8)

    # Nome do cluster (representante)
    cluster_names_initial = {cid: reps[cid]["uc"] for cid in range(k)}

    # --------- Projeção 2D ---------
    with st.spinner("🗺️ Projetando em 2D..."):
        xy = _project_2d(emb)
    df_plot = pd.DataFrame({"UC_ID": df_an["UC_ID"], "Nome da UC": nomes, "Cluster": labels, "X": xy[:, 0], "Y": xy[:, 1], "Ementa": textos})
    df_plot["Nome do Cluster (Rep.)"] = df_plot["Cluster"].map(cluster_names_initial)

    # --------- Tabela: UCs por Cluster ---------
    st.markdown("### 📋 UCs x Clusters (com UC_ID)")
    df_out = df_plot[["UC_ID", "Nome da UC", "Cluster", "Nome do Cluster (Rep.)", "Ementa"]].sort_values(["Cluster", "UC_ID"])
    st.dataframe(df_out, use_container_width=True, height=420)
    export_table(scope_key, df_out, "clusters_ucid", "Clusters_UCIDs")

    # --------- Palavras-chave por cluster ---------
    st.markdown("### 🔤 Palavras-chave (TF-IDF) por Cluster")
    if not df_kw.empty:
        df_kw["Nome do Cluster (Rep.)"] = df_kw["Cluster"].map(cluster_names_initial)
        df_kw["Palavras-chave (Top)"] = df_kw["Palavras-chave"].apply(lambda xs: ", ".join(xs) if xs else "—")
        st.dataframe(df_kw[["Cluster", "Nome do Cluster (Rep.)", "Palavras-chave (Top)"]], use_container_width=True)
        export_table(scope_key, df_kw[["Cluster", "Palavras-chave (Top)"]], "clusters_keywords", "Clusters_Keywords")
    else:
        st.info("Não foi possível extrair palavras-chave (TF-IDF) para os clusters.")

    # --------- Gráfico 1: rótulo baseado no representante ---------
    st.markdown("### 🗺️ Dispersão 2D (rótulo = UC representante)")
    _plot_scatter(
        df_plot.assign(_label=df_plot["Nome do Cluster (Rep.)"]),
        title="Distribuição (Ementa) — SBERT — rótulo por UC representante",
        label_col="_label",
        scope_key=scope_key,
        base_name="cluster_scatter_representante"
    )

    # --------- Nomear com GPT (opcional) ---------
    st.markdown("### 🤖 Nomear Clusters com GPT (opcional)")
    st.caption("Se desejar, forneça sua OpenAI API Key para o GPT sugerir nomes temáticos curtos para cada cluster.")
    api_key = st.text_input("🔑 OpenAI API Key", type="password")

    gpt_names = {}
    if api_key and st.button("🚀 Sugerir nomes com GPT"):
        # Prepara amostra compacta por cluster
        payload = []
        for cid in range(k):
            ex_idxs = np.where(labels == cid)[0]
            exemplo_ementa = textos[int(ex_idxs[0])] if ex_idxs.size > 0 else ""
            keywords = df_kw.loc[df_kw["Cluster"] == cid, "Palavras-chave"]
            keywords = keywords.iloc[0] if not keywords.empty else []
            payload.append({
                "cluster": int(cid),
                "representante": cluster_names_initial.get(cid, f"Cluster {cid}"),
                "keywords": keywords if isinstance(keywords, list) else [],
                "amostra_ementa": exemplo_ementa[:400]
            })

        with st.spinner("Consultando GPT para nomear clusters..."):
            gpt_names = _call_gpt_for_names(payload, api_key)

        if gpt_names:
            st.success("Nomes sugeridos recebidos!")
        else:
            st.warning("O GPT não retornou nomes. Mantendo rótulos pelos representantes.")

    # --------- Comparação Antes x Depois (se houver GPT) ---------
    if gpt_names:
        df_names = pd.DataFrame({
            "Cluster": list(range(k)),
            "Nome (Rep.)": [cluster_names_initial[c] for c in range(k)],
            "Nome GPT": [gpt_names.get(c, cluster_names_initial[c]) for c in range(k)]
        })
        st.markdown("### 🔁 Comparação de nomes (Antes × Depois)")
        st.dataframe(df_names, use_container_width=True)
        export_table(scope_key, df_names, "clusters_nomes_antes_depois", "Clusters_Nomes_Comparacao")

        # Gráfico 2 com rótulo GPT
        df_plot["Nome GPT"] = df_plot["Cluster"].map(lambda c: gpt_names.get(int(c), cluster_names_initial[int(c)]))
        st.markdown("### 🗺️ Dispersão 2D (rótulo = Nome GPT)")
        _plot_scatter(
            df_plot.assign(_label=df_plot["Nome GPT"]),
            title="Distribuição (Ementa) — SBERT — rótulo por Nome GPT",
            label_col="_label",
            scope_key=scope_key,
            base_name="cluster_scatter_gpt"
        )

    # --------- Tabela Consolidada (todas as UCs + nomes GPT, se existirem) ---------
    st.markdown("### 📋 Tabela Consolidada — Todas as UCs e Clusters")
    if gpt_names:
        df_full = df_out.merge(
            pd.DataFrame({
                "Cluster": list(range(k)),
                "Nome GPT": [gpt_names.get(c, cluster_names_initial[c]) for c in range(k)]
            }),
            on="Cluster",
            how="left"
        )
    else:
        df_full = df_out.copy()
        df_full["Nome GPT"] = "—"

    df_full = df_full[["UC_ID", "Nome da UC", "Cluster", "Nome do Cluster (Rep.)", "Nome GPT", "Ementa"]]
    st.dataframe(df_full, use_container_width=True, height=480)
    export_table(scope_key, df_full, "cluster_consolidado", "UCs e Clusters Consolidados")

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

        **4️⃣ Comparativo pós-GPT:**
        - O gráfico final evidencia como o GPT reorganizou semanticamente os clusters,
          ajudando a alinhar estatísticas a significados pedagógicos.

        **5️⃣ Aplicações práticas:**
        - Diagnóstico de **redundância e sobreposição curricular**.
        - Identificação de **áreas interdisciplinares** emergentes.
        - Planejamento de **integração entre clusters correlatos**.
        """
    )
