# ===============================================================
# 🔗 EmentaLabv2 — Grafo de Dependências Curriculares (v8.3)
# ===============================================================
from __future__ import annotations
import re
from typing import List, Tuple
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from openai import OpenAI
from utils.text_utils import find_col, truncate
from utils.exportkit import export_zip_button, export_table
from utils.embeddings import sbert_embed, l2_normalize
import numpy as np


# ---------------------------------------------------------------
# 1. Função auxiliar — parser textual
# ---------------------------------------------------------------
def _parse_dependencies(text: str) -> List[Tuple[str, str]]:
    """Extrai pares 'A -> B' ou frases equivalentes."""
    pairs = []
    if not text:
        return pairs

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        match_arrow = re.findall(r"(.+?)\s*[-–>]{1,2}\s*(.+)", ln)
        match_words = re.findall(
            r"(.+?)\s+(?:é|são)\s*pré[\-\s]?requisito[s]?\s+(?:de|para)\s+(.+)",
            ln,
            flags=re.IGNORECASE,
        )
        match_depends = re.findall(
            r"(.+?)\s+depende\s+(?:de|do)\s+(.+)", ln, flags=re.IGNORECASE
        )
        for a, b in match_arrow + match_words + match_depends:
            a, b = a.strip(" .,:;–-"), b.strip(" .,:;–-")
            if a and b and a != b:
                pairs.append((a, b))

    # Remover duplicatas
    seen = set()
    clean = []
    for a, b in pairs:
        key = (a.lower(), b.lower())
        if key not in seen:
            seen.add(key)
            clean.append((a, b))
    return clean


# ---------------------------------------------------------------
# 2. Fallback SBERT — gera dependências por similaridade
# ---------------------------------------------------------------
def _infer_semantic_links(df: pd.DataFrame, col_text: str, n_top: int = 2) -> List[Tuple[str, str]]:
    """Cria pares de pré-requisito prováveis com base em similaridade SBERT."""
    nomes = df["Nome da UC"].astype(str).tolist()
    textos = df[col_text].astype(str).tolist()
    if len(textos) < 2:
        return []

    emb = l2_normalize(sbert_embed(textos))
    sims = np.dot(emb, emb.T)

    pairs = []
    for i, nome_a in enumerate(nomes):
        idx_sorted = np.argsort(-sims[i])
        for j in idx_sorted[1 : n_top + 1]:
            if sims[i, j] > 0.45:  # limiar semântico
                pairs.append((nome_a, nomes[j]))
    # remove duplicatas e reflexivos
    pairs = list({(a, b) for a, b in pairs if a != b})
    return pairs


# ---------------------------------------------------------------
# 3. Desenha grafo
# ---------------------------------------------------------------
def _draw_graph(pairs: List[Tuple[str, str]]) -> plt.Figure:
    G = nx.DiGraph()
    G.add_edges_from(pairs)
    pos = nx.spring_layout(G, seed=42, k=0.8)

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color="#a5d8ff", node_size=1600, alpha=0.95, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#3b5bdb", arrows=True, arrowsize=18, width=2.0, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_color="#111", ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------
# 4. Função principal
# ---------------------------------------------------------------
def run_graph(df: pd.DataFrame, scope_key: str) -> None:
    st.header("🔗 Dependência Curricular — Relações de Pré-requisito entre UCs")
    st.caption(
        "Identifica precedências entre UCs com base nos conteúdos programáticos. "
        "Usa GPT para inferência textual e SBERT como fallback semântico."
    )

    # Localiza coluna de conteúdo
    col_obj = find_col(df, "Objetos de conhecimento") or find_col(df, "Conteúdo programático")
    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' (ou 'Conteúdo programático') não encontrada.")
        return

    # Subconjunto de UCs
    subset = df[["Nome da UC", col_obj]].dropna()
    if subset.empty:
        st.warning("Nenhuma UC com 'Objetos de conhecimento' preenchido.")
        return

    max_uc = st.slider(
        "Quantidade de UCs a considerar (amostra para análise GPT)",
        min_value=4,
        max_value=min(40, len(subset)),
        value=min(12, len(subset)),
        step=1,
    )
    subset = subset.head(max_uc)

    # Chave API
    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    use_fallback = st.checkbox("⚙️ Ativar fallback automático por similaridade SBERT", value=True)
    if not api_key:
        st.info("Informe a OpenAI API Key para executar a análise GPT.")
        return

    client = OpenAI(api_key=api_key)

    # Prompt mais direto
    prompt_lines = [
        "Você deve OBRIGATORIAMENTE indicar relações diretas de pré-requisito entre as UCs listadas.",
        "Responda APENAS no formato 'A -> B', onde A é pré-requisito de B.",
        "Não descreva as UCs individualmente. Se não houver relação, ignore a UC.",
        "",
        "Exemplo:",
        "- Expressão e Linguagens Visuais -> Meios de Representação",
        "- Meios de Representação -> Projeto de Ambientes e Interiores Residenciais",
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
            temperature=0.0,
        )
    content = (resp.choices[0].message.content or "").strip()

    st.subheader("📄 Saída textual do modelo (para auditoria)")
    st.text_area("Diagnóstico do modelo", value=content, height=250)

    # -----------------------------------------------------------
    # Parsing do texto
    # -----------------------------------------------------------
    pairs_gpt = _parse_dependencies(content)
    pairs = pairs_gpt.copy()

    if not pairs_gpt and use_fallback:
        st.warning("⚠️ Nenhuma relação explícita detectada. Usando fallback SBERT…")
        pairs = _infer_semantic_links(subset, col_obj)

    if not pairs:
        st.error("❌ Nenhuma relação de pré-requisito foi identificada (nem pelo GPT nem por similaridade).")
        export_zip_button(scope_key)
        return

    # -----------------------------------------------------------
    # Visualização e exportação
    # -----------------------------------------------------------
    st.subheader("🌐 Grafo de Pré-requisitos")
    fig = _draw_graph(pairs)
    st.pyplot(fig)

    df_edges = pd.DataFrame(pairs, columns=["Pré-requisito", "UC Dependente"])
    export_table(scope_key, df_edges, "grafo_pre_requisitos", "Relações de Pré-requisito")
    export_zip_button(scope_key)

    # Resumo numérico
    st.markdown("---")
    st.metric("Número de UCs analisadas", len(subset))
    st.metric("Relações identificadas", len(pairs))
