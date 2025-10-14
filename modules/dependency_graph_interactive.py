# ===============================================================
# 🔗 EmentaLabv2 — Grafo Interativo de Dependências (v9.0)
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
from streamlit_pyvis import st_pyvis
import tempfile, os

# ---------------------------------------------------------------
# 🔍 Funções auxiliares
# ---------------------------------------------------------------
def _parse_dependencies_with_reasons(text: str) -> List[Tuple[str, str, str]]:
    """
    Extrai pares 'A -> B' e tenta capturar justificativas.
    Exemplo esperado no texto:
    'A -> B: porque A fornece os fundamentos de X usados em B.'
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


def _infer_semantic_links(df: pd.DataFrame, col_text: str, n_top: int = 2) -> List[Tuple[str, str, str]]:
    """Cria pares prováveis com base em similaridade SBERT (fallback automático)."""
    nomes = df["Nome da UC"].astype(str).tolist()
    textos = df[col_text].astype(str).tolist()
    if len(textos) < 2:
        return []
    emb = l2_normalize(sbert_embed(textos))
    sims = np.dot(emb, emb.T)
    triples = []
    for i, nome_a in enumerate(nomes):
        idx_sorted = np.argsort(-sims[i])
        for j in idx_sorted[1 : n_top + 1]:
            if sims[i, j] > 0.45:
                reason = f"Similaridade semântica de {sims[i,j]:.2f} entre os conteúdos de {nome_a} e {nomes[j]}"
                triples.append((nome_a, nomes[j], reason))
    triples = list({(a, b, r) for a, b, r in triples if a != b})
    return triples


def _draw_pyvis_graph(pairs: List[Tuple[str, str, str]]) -> Network:
    """Cria grafo PyVis interativo diretamente embutido no Streamlit."""
    nt = Network(
        height="700px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#1c1c1c",
        directed=True,
        notebook=False,
    )

    # Layout hierárquico para evitar sobreposição
    nt.set_options("""
        const options = {
            "layout": {"hierarchical": {
                "enabled": true,
                "direction": "LR",
                "sortMethod": "hubsize"
            }},
            "edges": {
                "color": {"inherit": false},
                "smooth": false,
                "arrows": {"to": {"enabled": true}}
            },
            "nodes": {
                "shape": "box",
                "font": {"size": 14, "multi": "html"},
                "color": {"background": "#e7f5ff", "border": "#1c7ed6"}
            },
            "physics": false
        }
    """)

    for a, b, reason in pairs:
        nt.add_node(a, label=a)
        nt.add_node(b, label=b)
        nt.add_edge(a, b, title=reason, color="#228be6")

    return nt


# ---------------------------------------------------------------
# 🚀 Função principal
# ---------------------------------------------------------------
def run_graph_interactive(df: pd.DataFrame, scope_key: str):
    st.header("🌐 Grafo Interativo — Relações de Pré-requisito entre UCs")
    st.markdown("""
    Este módulo identifica **relações de dependência e precedência** entre as **Unidades Curriculares (UCs)**,
    revelando como o conhecimento se encadeia ao longo da matriz curricular.
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

    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    use_fallback = st.checkbox("⚙️ Ativar fallback automático SBERT", value=True)
    if not api_key:
        st.info("Informe a OpenAI API Key para executar a análise GPT.")
        return

    client = OpenAI(api_key=api_key)

    # ---------------- GPT Prompt ----------------
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

    # ---------------- Execução GPT ----------------
    with st.spinner("🧠 Gerando análise via GPT..."):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
    content = (resp.choices[0].message.content or "").strip()

    st.subheader("📄 Saída textual do modelo (auditoria)")
    st.text_area("Diagnóstico do modelo", value=content, height=200)

    triples = _parse_dependencies_with_reasons(content)
    if not triples and use_fallback:
        st.warning("⚠️ Nenhuma relação explícita detectada. Usando fallback SBERT…")
        triples = _infer_semantic_links(subset, col_obj)

    if not triples:
        st.error("❌ Nenhuma relação identificada (nem GPT nem SBERT).")
        export_zip_button(scope_key)
        return

    # ---------------- Grafo interativo ----------------
    st.markdown("### 🌍 Visualização Interativa")
    graph = _draw_pyvis_graph(triples)
    st_pyvis(graph, height="700px")

    # ---------------- Tabela explicativa ----------------
    df_edges = pd.DataFrame(triples, columns=["UC (Pré-requisito)", "UC Dependente", "Justificativa"])
    st.markdown("### 📘 Relações Identificadas e Justificativas")
    st.dataframe(df_edges, use_container_width=True, hide_index=True)

    export_table(scope_key, df_edges, "grafo_interativo_pre_requisitos_com_justificativas", "Relações Pré-requisito com Justificativas")
    export_zip_button(scope_key)

    # ---------------- Métricas rápidas ----------------
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("UCs analisadas", len(subset))
    c2.metric("Relações identificadas", len(triples))

    # ---------------- Painel interpretativo ----------------
    with st.expander("🧭 Como interpretar o grafo e aplicar os resultados", expanded=False):
        st.markdown("""
        ### 🔹 1. Interpretação visual
        - Cada **nó** é uma UC; cada **seta** indica uma dependência.
        - Passe o mouse sobre a seta para ver a **justificativa textual do GPT**.
        - **Nós de origem (com muitas saídas)** representam **disciplinas-fundamento**.
        - **Nós de destino (com muitas entradas)** indicam **sínteses e projetos integradores**.

        ### 🔹 2. Como usar a tabela
        A tabela logo acima lista cada relação detectada e a explicação do GPT.
        - Serve como **trilha de auditoria pedagógica**, permitindo verificar se as inferências fazem sentido.
        - Pode ser exportada para **documentos de NDE/PPC**.

        ### 🔹 3. Aplicações práticas
        - Identificar **cadeias formativas coerentes** (Fundamentos → Aplicações → Síntese);
        - Detectar **lacunas** (UCs isoladas, sem pré-requisitos ou dependentes);
        - Revisar **sequência lógica de competências** no currículo;
        - Apoiar **evidências de coerência vertical** no ENADE e nas avaliações do MEC.
        """)
