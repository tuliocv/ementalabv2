# ===============================================================
# 🔗 EmentaLabv2 — Grafo Interativo de Dependências (v8.4)
# ===============================================================
import re
from typing import List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from utils.text_utils import find_col, truncate
from utils.exportkit import export_table, export_zip_button
from utils.embeddings import sbert_embed, l2_normalize
from pyvis.network import Network
import tempfile, os


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
            ln, flags=re.IGNORECASE,
        )
        match_depends = re.findall(
            r"(.+?)\s+depende\s+(?:de|do)\s+(.+)", ln, flags=re.IGNORECASE
        )
        for a, b in match_arrow + match_words + match_depends:
            a, b = a.strip(" .,:;–-"), b.strip(" .,:;–-")
            if a and b and a != b:
                pairs.append((a, b))
    seen = set()
    clean = []
    for a, b in pairs:
        key = (a.lower(), b.lower())
        if key not in seen:
            seen.add(key)
            clean.append((a, b))
    return clean


def _infer_semantic_links(df: pd.DataFrame, col_text: str, n_top: int = 2) -> List[Tuple[str, str]]:
    """Cria pares prováveis com base em similaridade SBERT."""
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
            if sims[i, j] > 0.45:
                pairs.append((nome_a, nomes[j]))
    pairs = list({(a, b) for a, b in pairs if a != b})
    return pairs


def _draw_interactive_graph(pairs: List[Tuple[str, str]]) -> str:
    """Gera grafo interativo e retorna caminho HTML temporário."""
    nt = Network(height="650px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
    nt.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=150, spring_strength=0.02)
    for a, b in pairs:
        nt.add_node(a, label=a, color="#a5d8ff")
        nt.add_node(b, label=b, color="#74c0fc")
        nt.add_edge(a, b, color="#1c7ed6", arrowStrikethrough=False)
    nt.repulsion(node_distance=180, spring_length=150)
    tmp_path = os.path.join(tempfile.gettempdir(), "grafo_interativo.html")
    nt.save_graph(tmp_path)
    return tmp_path


def run_graph_interactive(df: pd.DataFrame, scope_key: str):
    st.header("🌐 Grafo Interativo — Relações de Pré-requisito entre UCs")
    st.caption(
        "Identifica precedências entre UCs com base nos conteúdos programáticos. "
        "Usa GPT para inferência textual e SBERT como fallback semântico, exibindo grafo interativo em HTML."
    )

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

    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    use_fallback = st.checkbox("⚙️ Ativar fallback automático SBERT", value=True)
    if not api_key:
        st.info("Informe a OpenAI API Key para executar a análise GPT.")
        return

    client = OpenAI(api_key=api_key)

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

    pairs = _parse_dependencies(content)
    if not pairs and use_fallback:
        st.warning("⚠️ Nenhuma relação explícita detectada. Usando fallback SBERT…")
        pairs = _infer_semantic_links(subset, col_obj)

    if not pairs:
        st.error("❌ Nenhuma relação identificada (nem GPT nem SBERT).")
        export_zip_button(scope_key)
        return

    html_path = _draw_interactive_graph(pairs)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=700, scrolling=True)

    df_edges = pd.DataFrame(pairs, columns=["Pré-requisito", "UC Dependente"])
    export_table(scope_key, df_edges, "grafo_interativo_pre_requisitos", "Relações Pré-requisito (Interativo)")
    export_zip_button(scope_key)

    st.markdown("---")
    st.metric("UCs analisadas", len(subset))
    st.metric("Relações identificadas", len(pairs))
