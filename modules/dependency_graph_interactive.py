# ===============================================================
# 🔗 EmentaLabv2 — Grafo de Dependências (v9.2 - Estático)
# ===============================================================
import re
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
from utils.text_utils import find_col, truncate
from utils.embeddings import sbert_embed, l2_normalize
from utils.exportkit import export_table, export_zip_button


# ---------------------------------------------------------------
# 🔍 Extração de relações
# ---------------------------------------------------------------
def _parse_dependencies_with_reasons(text: str):
    """
    Extrai pares 'A -> B' e justificativas (quando houver).
    Exemplo: "A -> B: porque A fornece base teórica para B"
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    triples = []
    pattern = re.compile(r"(.+?)\s*[-–>]{1,2}\s*(.+?)(?::\s*(.+))?$")
    for ln in lines:
        match = pattern.match(ln)
        if match:
            a, b, reason = match.groups()
            a, b = a.strip(" .,:;–-"), b.strip(" .,:;–-")
            if a and b and a != b:
                triples.append((a, b, reason or "—"))
    seen = set()
    clean = []
    for a, b, r in triples:
        key = (a.lower(), b.lower())
        if key not in seen:
            seen.add(key)
            clean.append((a, b, r))
    return clean


def _infer_semantic_links(df, col_text, n_top=2):
    """Fallback automático com SBERT (sem GPT)."""
    nomes = df["Nome da UC"].astype(str).tolist()
    textos = df[col_text].astype(str).tolist()
    if len(textos) < 2:
        return []
    emb = l2_normalize(sbert_embed(textos))
    sims = np.dot(emb, emb.T)
    triples = []
    for i, nome_a in enumerate(nomes):
        idx_sorted = np.argsort(-sims[i])
        for j in idx_sorted[1:n_top+1]:
            if sims[i, j] > 0.45:
                reason = f"Similaridade semântica de {sims[i,j]:.2f} entre conteúdos de {nome_a} e {nomes[j]}"
                triples.append((nome_a, nomes[j], reason))
    triples = list({(a, b, r) for a, b, r in triples if a != b})
    return triples


# ---------------------------------------------------------------
# 🎨 Desenho do grafo (estático, organizado e legível)
# ---------------------------------------------------------------
def _draw_static_graph(pairs):
    """Desenha o grafo com layout hierárquico e melhor espaçamento."""
    if not pairs:
        return None

    G = nx.DiGraph()
    for a, b, _ in pairs:
        G.add_edge(a, b)

    # Layout hierárquico orientado (da esquerda para a direita)
    pos = nx.multipartite_layout(G, subset_key=lambda n: nx.shortest_path_length(G, list(G.nodes)[0], n)
                                 if nx.has_path(G, list(G.nodes)[0], n) else 0)

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=1800, node_color="#a5d8ff", edgecolors="#1c7ed6")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=18, edge_color="#1c7ed6", width=2, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_color="#1c1c1c")

    plt.title("Mapa de Dependências entre UCs", fontsize=14, fontweight="bold", pad=20)
    plt.axis("off")
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()


# ---------------------------------------------------------------
# 🚀 Função principal
# ---------------------------------------------------------------
def run_graph_interactive(df, scope_key):
    st.header("🌐 Mapa de Dependências — Relações de Pré-requisito entre UCs")
    st.markdown("""
    Este módulo identifica **relações de dependência e precedência** entre as Unidades Curriculares (UCs),
    com base nos **objetos de conhecimento** e **conteúdos programáticos**.
    O resultado é um **gráfico estático hierárquico**, que mostra o fluxo de aprendizagem ao longo da matriz curricular.
    """)

    col_obj = find_col(df, "Objetos de conhecimento") or find_col(df, "Conteúdo programático")
    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' (ou 'Conteúdo programático') não encontrada.")
        return

    subset = df[["Nome da UC", col_obj]].dropna()
    if subset.empty:
        st.warning("Nenhuma UC com 'Objetos de conhecimento' preenchido.")
        return

    max_uc = st.slider("Quantidade de UCs (amostra GPT)", 4, min(40, len(subset)), min(12, len(subset)), 1)
    subset = subset.head(max_uc)

    api_key = st.text_input("🔑 OpenAI API Key (opcional, para inferência GPT)", type="password")
    use_fallback = st.checkbox("⚙️ Ativar fallback automático SBERT", value=True)

    triples = []
    if api_key:
        client = OpenAI(api_key=api_key)
        prompt_lines = [
            "Você deve indicar relações diretas de pré-requisito entre as UCs listadas.",
            "Responda no formato 'A -> B: justificativa'.",
            "Exemplo:",
            "Expressão e Linguagens Visuais -> Meios de Representação: fornece fundamentos visuais necessários para representação técnica.",
            "",
            "UCs (nome: objetos de conhecimento):",
        ]
        for _, r in subset.iterrows():
            prompt_lines.append(f"- {r['Nome da UC']}: {truncate(str(r[col_obj]), 600)}")
        prompt = "\n".join(prompt_lines)

        with st.spinner("🧠 Gerando análise via GPT..."):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
        content = (resp.choices[0].message.content or "").strip()
        triples = _parse_dependencies_with_reasons(content)

    if not triples and use_fallback:
        st.warning("⚠️ Nenhuma relação explícita detectada. Usando fallback SBERT…")
        triples = _infer_semantic_links(subset, col_obj)

    if not triples:
        st.error("❌ Nenhuma relação identificada (nem GPT nem SBERT).")
        export_zip_button(scope_key)
        return

    # ---------------- Gráfico ----------------
    st.markdown("### 🎨 Mapa de Dependências entre UCs")
    _draw_static_graph(triples)

    # ---------------- Tabela ----------------
    df_edges = pd.DataFrame(triples, columns=["UC (Pré-requisito)", "UC Dependente", "Justificativa"])
    st.markdown("### 📘 Relações Identificadas e Justificativas")
    st.dataframe(df_edges, use_container_width=True, hide_index=True)
    export_table(scope_key, df_edges, "grafo_estatico_pre_requisitos", "Relações Pré-requisito")
    export_zip_button(scope_key)

    # ---------------- Métricas ----------------
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("UCs analisadas", len(subset))
    c2.metric("Relações identificadas", len(triples))

    with st.expander("🧭 Como interpretar o gráfico", expanded=False):
        st.markdown("""
        ### 🔹 Leitura do Mapa
        - Cada **nó** representa uma UC.
        - Cada **seta** indica uma **relação de dependência** (A → B = A é pré-requisito de B).
        - O grafo é desenhado da **esquerda para a direita**, mostrando o avanço formativo.
        - UCs mais à esquerda são **fundamentais**, e as mais à direita **dependem de múltiplas bases**.

        ### 🔹 Análises Possíveis
        - **Coerência vertical**: se as UCs seguem uma sequência lógica de complexidade crescente.
        - **Lacunas**: UCs isoladas sem ligações (podem indicar desconexões no currículo).
        - **Densidade de conexões**: alto número de setas indica forte integração interdisciplinar.
        """)
