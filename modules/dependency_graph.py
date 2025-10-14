# ementalabv2/modules/dependency_graph.py
# ===============================================================
# Sequenciamento / Grafo (GPT + Visual) — versão limpa (Py3.10)
# ===============================================================
from __future__ import annotations

import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Deixa mensagem amigável se lib não existir

from utils.text_utils import find_col, truncate
from utils.exportkit import export_zip_button, export_table


def _parse_dependencies(text: str) -> List[Tuple[str, str]]:
    """
    Extrai pares (A -> B) de pré-requisito a partir do texto do GPT.
    Padrões suportados (case-insensitive):
      - 'X é pré-requisito de Y'
      - 'X é pré requisito para Y'
      - 'Y depende de X'
      - '- X -> Y' (flecha)
    Retorna lista de tuplas (pre_req, dependente).
    """
    pairs: List[Tuple[str, str]] = []
    if not text:
        return pairs

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        # 1) X é pré-requisito de/para Y
        m1 = re.findall(
            r"(.+?)\s+(?:e|é|são)?\s*pré[\-\s]?requisito[s]?\s+(?:de|para)\s+(.+)",
            ln,
            flags=re.IGNORECASE,
        )
        # 2) Y depende de X
        m2 = re.findall(
            r"(.+?)\s+depende\s+(?:de|do)\s+(.+)",
            ln,
            flags=re.IGNORECASE,
        )
        # 3) Lista com flecha: X -> Y
        m3 = re.findall(
            r"^\-?\s*([^:\-–>]+?)\s*[-–>]{1,2}\s*(.+)$",  # suporta '-', '->', '–>'
            ln,
            flags=re.IGNORECASE,
        )

        for a, b in (m1 + [(b, a) for (b, a) in m2] + m3):
            a = a.strip(" .,:;–-")
            b = b.strip(" .,:;–-")
            if a and b and a != b:
                # Evita linhas gigantes/ruído
                if len(a) <= 120 and len(b) <= 120:
                    pairs.append((a, b))

    # Dedup
    seen = set()
    out = []
    for a, b in pairs:
        key = (a.lower(), b.lower())
        if key not in seen:
            seen.add(key)
            out.append((a, b))
    return out


def _draw_graph(pairs: List[Tuple[str, str]]) -> plt.Figure:
    """Desenha grafo simples com NetworkX e Matplotlib."""
    G = nx.DiGraph()
    G.add_edges_from(pairs)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.8)

    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw_networkx_nodes(G, pos, node_color="#a5d8ff", node_size=1600, alpha=0.95, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#3b5bdb", arrows=True, arrowsize=18, width=2.0, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_color="#111", ax=ax)

    ax.set_axis_off()
    fig.tight_layout()
    return fig


def run_graph(df: pd.DataFrame, scope_key: str) -> None:
    """
    Entrada principal do módulo (chamada pelo app).
    - Usa GPT para gerar um diagnóstico textual das dependências.
    - Faz parsing do texto para pares A->B.
    - Gera grafo e permite exportar CSV/ZIP.
    """
    st.header("🔗 Dependência Curricular — Relações de Pré-requisito entre UCs")
    st.caption(
        "Identifica precedências entre UCs com base nos conteúdos programáticos. "
        "Gera uma leitura textual via GPT e, em seguida, um grafo dirigido."
    )

    # Verifica coluna de objetos/conteúdo
    col_obj = find_col(df, "Objetos de conhecimento") or find_col(df, "Conteúdo programático")
    if not col_obj:
        st.error("Coluna de 'Objetos de conhecimento' (ou 'Conteúdo programático') não encontrada.")
        return

    # Subconjunto (para evitar prompts gigantes)
    subset = df[["Nome da UC", col_obj]].dropna()
    if subset.empty:
        st.warning("Não há UCs válidas com 'Objetos de conhecimento' preenchidos.")
        return

    max_uc = st.slider(
        "Quantidade de UCs a considerar no prompt (amostra)",
        min_value=4,
        max_value=min(40, len(subset)),
        value=min(12, len(subset)),
        step=1,
    )
    subset = subset.head(max_uc).copy()

    # API Key
    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    if not api_key:
        st.info("Informe a OpenAI API Key para prosseguir.")
        return
    if OpenAI is None:
        st.error("Pacote 'openai' não está instalado. Adicione 'openai' ao requirements.txt.")
        return

    # Prompt
    prompt_lines = [
        "Analise as UCs abaixo e identifique as relações de pré-requisito entre elas.",
        "Ao listar, use frases simples como:",
        "- X é pré-requisito de Y",
        "- Y depende de X",
        "Se fizer sentido, você também pode usar 'X -> Y'.",
        "",
        "UCs (nome: objetos de conhecimento):",
    ]
    for _, r in subset.iterrows():
        prompt_lines.append(f"- {r['Nome da UC']}: {truncate(str(r[col_obj]).replace(';', ', '), 700)}")

    prompt = "\n".join(prompt_lines)

    # Chamada ao modelo
    client = OpenAI(api_key=api_key)
    with st.spinner("Consultando o modelo GPT..."):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    content = (resp.choices[0].message.content or "").strip()

    st.subheader("📄 Saída textual do modelo (para auditoria)")
    st.text_area("Diagnóstico do modelo", value=content, height=260)

    # Parsing de dependências
    pairs_all = _parse_dependencies(content)

    # Filtro: manter apenas UCs presentes no subset (para evitar nomes “fantasma”)
    uc_names = set(subset["Nome da UC"].astype(str))
    pairs = [(a, b) for (a, b) in pairs_all if a in uc_names and b in uc_names]

    if not pairs:
        st.warning("Nenhum padrão de pré-requisito foi identificado na resposta do GPT.")
        export_zip_button(scope_key)
        return

    # Visualização + Export
    st.subheader("🌐 Grafo de Pré-requisitos (A → B)")
    fig = _draw_graph(pairs)
    st.pyplot(fig)

    df_edges = pd.DataFrame(pairs, columns=["Pré-requisito", "UC Dependente"])
    export_table(scope_key, df_edges, "grafo_pre_requisitos", "Relacoes_Pre_Requisitos")

    export_zip_button(scope_key)
