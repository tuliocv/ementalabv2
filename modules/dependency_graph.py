# ===============================================================
# 🔗 EmentaLabv2 — Grafo de Dependências (v11.3 — Estático + Análise)
# ===============================================================
# - Extrai relações A -> B + justificativas do GPT (opcional)
# - Fallback SBERT quando GPT não retorna relações
# - Desenha grafo estático hierárquico (menos sobreposição)
# - Gera tabelas amigáveis (pré-requisitos por UC e “esta UC prepara para”)
# - Produz análise interpretativa automática (pontos fortes e fracos)
# - Exporta figuras e tabelas em ZIP do escopo
# ===============================================================

from __future__ import annotations
import re
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

# GPT (opcional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # para evitar import error em ambientes sem openai

# Utils do projeto
from utils.text_utils import find_col, truncate
from utils.embeddings import sbert_embed, l2_normalize
from utils.exportkit import export_table, show_and_export_fig, export_zip_button


# ---------------------------------------------------------------
# 🔍 Parsers e heurísticas
# ---------------------------------------------------------------
def _parse_dependencies_with_reasons(text: str) -> List[Tuple[str, str, str]]:
    """
    Extrai pares 'A -> B: justificativa' (justificativa opcional).
    Linhas válidas:
      - A -> B
      - A -> B: porque ...
      - A –> B: ...
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    triples = []
    pattern = re.compile(r"(.+?)\s*[-–>]{1,2}\s*(.+?)(?::\s*(.+))?$")
    for ln in lines:
        m = pattern.match(ln)
        if not m:
            continue
        a, b, reason = m.groups()
        a, b = a.strip(" .,:;–-"), b.strip(" .,:;–-")
        if a and b and a != b:
            triples.append((a, b, (reason or "—").strip()))
    # de-duplicação simples
    seen = set()
    clean = []
    for a, b, r in triples:
        key = (a.lower(), b.lower())
        if key not in seen:
            seen.add(key)
            clean.append((a, b, r))
    return clean


def _infer_semantic_links(df: pd.DataFrame, col_text: str, n_top: int = 2, thr: float = 0.45) -> List[Tuple[str, str, str]]:
    """Fallback com SBERT: cria pares prováveis com pequena justificativa baseada em similaridade."""
    nomes = df["Nome da UC"].astype(str).tolist()
    textos = df[col_text].astype(str).tolist()
    if len(textos) < 2:
        return []
    emb = l2_normalize(sbert_embed(textos))
    sims = np.dot(emb, emb.T)
    triples = []
    for i, nome_a in enumerate(nomes):
        idx_sorted = np.argsort(-sims[i])
        for j in idx_sorted[1:n_top + 1]:
            if sims[i, j] > thr:
                reason = f"Similaridade semântica {sims[i,j]:.2f} entre conteúdos de '{nome_a}' e '{nomes[j]}'"
                triples.append((nome_a, nomes[j], reason))
    # remover auto-loops/duplicados
    triples = list({(a, b, r) for a, b, r in triples if a != b})
    return triples


# ---------------------------------------------------------------
# 📐 Layout hierárquico estável
# ---------------------------------------------------------------
def _compute_layers(pairs: List[Tuple[str, str, str]]) -> Dict[str, int]:
    """
    Atribui camadas (níveis) para cada nó com base em pré-requisitos:
      - nós sem pré-requisito direto ficam na camada 0;
      - destino recebe camada >= (max(camadas de seus pré) + 1).
    Se houver ciclos, aplica camadas aproximadas por indegree BFS.
    """
    G = nx.DiGraph()
    for a, b, _ in pairs:
        G.add_edge(a, b)

    # tenta topological layering
    levels = {}
    try:
        order = list(nx.topological_sort(G))
        for node in order:
            preds = list(G.predecessors(node))
            if not preds:
                levels[node] = 0
            else:
                levels[node] = max(levels[p] for p in preds) + 1
        return levels
    except Exception:
        # fallback (grafos com ciclo): BFS por indegree
        indeg = dict(G.in_degree())
        q = [n for n, d in indeg.items() if d == 0]
        levels = {n: 0 for n in q}
        visited = set(q)
        while q:
            cur = q.pop(0)
            for nxt in G.successors(cur):
                if nxt not in visited:
                    levels[nxt] = max(levels.get(nxt, 0), levels[cur] + 1)
                    q.append(nxt)
                    visited.add(nxt)
        # nós restantes (cíclicos) → última camada conhecida + 1
        max_layer = max(levels.values()) if levels else 0
        for n in G.nodes:
            if n not in levels:
                levels[n] = max_layer + 1
        return levels


def _layout_by_layers(levels: Dict[str, int], G: nx.DiGraph) -> Dict[str, tuple]:
    """
    Constrói posições (x, y) usando camadas em Y e nós espaçados em X.
    Garante margens e ordenação estável.
    """
    # agrupa por camada
    by_layer = {}
    for n, l in levels.items():
        by_layer.setdefault(l, []).append(n)

    # ordena alfabeticamente dentro da camada para estabilidade
    for l in by_layer:
        by_layer[l].sort()

    # parâmetros de layout
    layer_gap_y = 1.8
    node_gap_x = 1.4

    pos = {}
    for layer, nodes in by_layer.items():
        width = (len(nodes) - 1) * node_gap_x
        start_x = -width / 2.0
        y = -layer * layer_gap_y
        for i, n in enumerate(nodes):
            pos[n] = (start_x + i * node_gap_x, y)
    return pos


def _draw_static_graph(triples: List[Tuple[str, str, str]], title: str = "Mapa de Dependências entre UCs"):
    """Desenha o grafo (estático) com layout hierárquico calculado."""
    if not triples:
        return None

    G = nx.DiGraph()
    for a, b, _ in triples:
        G.add_edge(a, b)

    # camadas & layout
    levels = _compute_layers(triples)
    pos = _layout_by_layers(levels, G)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=1800, node_color="#E3F2FD", edgecolors="#1976D2", linewidths=1.2, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=18, edge_color="#1976D2", width=2.0, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_color="#1c1c1c", ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    ax.axis("off")
    return fig, levels


# ---------------------------------------------------------------
# 🧾 Análise interpretativa automática
# ---------------------------------------------------------------
def _analysis_text(triples: List[Tuple[str, str, str]]) -> str:
    if not triples:
        return "Não foram identificadas relações de pré-requisito."

    df_edges = pd.DataFrame(triples, columns=["UC_Base", "UC_Destino", "Justificativa"])
    todas_ucs = sorted(set(df_edges["UC_Base"]).union(set(df_edges["UC_Destino"])))

    deps_por_base = df_edges.groupby("UC_Base")["UC_Destino"].nunique()
    bases_por_dest = df_edges.groupby("UC_Destino")["UC_Base"].nunique()

    uc_influente = deps_por_base.idxmax() if not deps_por_base.empty else None
    uc_dependente = bases_por_dest.idxmax() if not bases_por_dest.empty else None

    isoladas = [uc for uc in todas_ucs if uc not in deps_por_base.index and uc not in bases_por_dest.index]

    n = len(todas_ucs)
    m = len(df_edges)
    densidade = m / (n * (n - 1)) if n > 1 else 0.0

    parts = []
    parts.append(f"A análise identificou **{m} relações** entre **{n} UCs**, com densidade média de **{densidade:.2f}** no grafo.")

    if uc_influente:
        parts.append(f"A UC **{uc_influente}** destaca-se como **base estruturante** (fornece fundamentos para várias outras).")

    if uc_dependente:
        parts.append(f"A UC **{uc_dependente}** apresenta **alta interdependência**, exigindo múltiplos pré-requisitos.")

    if isoladas:
        parts.append(f"Foram encontradas **{len(isoladas)} UCs isoladas** (sem conexões diretas): {', '.join(isoladas)}.")

    if densidade > 0.35:
        parts.append("O alto índice de conexões sugere **forte articulação curricular** e progressão consistente.")
    elif densidade > 0.15:
        parts.append("Há **equilíbrio** entre autonomia e integração das UCs, com boa coerência vertical.")
    else:
        parts.append("A **baixa densidade** indica possível **fragmentação** e oportunidades de reforçar vínculos entre UCs.")

    parts.append("Recomenda-se revisar os encadeamentos nas UCs isoladas e verificar se as justificativas de pré-requisito refletem **competências e objetos** previstos no PPC.")

    return "\n\n".join(f"✅ {p}" for p in parts)


# ---------------------------------------------------------------
# 🚀 Função principal (assinatura usada no app: run_graph(df, scope_key))
# ---------------------------------------------------------------
def run_graph(df: pd.DataFrame, scope_key: str):
    st.header("🔗 Dependência Curricular")
    st.caption(
        "Identifica relações de **pré-requisito (A → B)** e interdependência entre UCs a partir dos "
        "**Objetos de Conhecimento / Conteúdos Programáticos**. "
        "Você pode usar **GPT** para justificar as ligações ou ativar o **fallback semântico (SBERT)**."
    )

    col_obj = find_col(df, "Objetos de conhecimento") or find_col(df, "Conteúdo programático")
    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' (ou 'Conteúdo programático') não encontrada.")
        return

    base = df[["Nome da UC", col_obj]].dropna().copy()
    if base.empty:
        st.warning("Nenhuma UC com 'Objetos de conhecimento' preenchido.")
        return

    # Amostragem (para prompts mais leves)
    max_uc = st.slider("Quantidade de UCs a considerar (amostra)", 4, min(50, len(base)), min(14, len(base)), 1)
    subset = base.head(max_uc).reset_index(drop=True)

    # Chave (tenta reaproveitar global; senão, campo local)
    api_key = st.session_state.get("global_api_key", "")
    api_key = st.text_input("🔑 OpenAI API Key (opcional para justificativas GPT)", value=api_key, type="password")
    if api_key:
        st.session_state["global_api_key"] = api_key

    use_fallback = st.checkbox("⚙️ Ativar fallback automático SBERT (se GPT não retornar ligações)", value=True)

    triples: List[Tuple[str, str, str]] = []

    # ---------------- GPT (opcional) ----------------
    if api_key and OpenAI is not None:
        try:
            client = OpenAI(api_key=api_key)
            # prompt compacto e formatado
            prompt_lines = [
                "TAREFA: indique **apenas** relações de pré-requisito **diretas** entre as UCs a seguir.",
                "FORMATO OBRIGATÓRIO (uma por linha): A -> B: justificativa curta",
                "Regra: A é pré-requisito de B quando o conteúdo de A é necessário para cursar B.",
                "",
                "EXEMPLO:",
                "Expressão e Linguagens Visuais -> Meios de Representação: fundamentos visuais para representação técnica",
                "Fundamentos de Cálculo -> Cálculo I: base conceitual de limites e derivadas",
                "",
                "UCs (nome: objetos/conteúdos):",
            ]
            for _, r in subset.iterrows():
                prompt_lines.append(f"- {r['Nome da UC']}: {truncate(str(r[col_obj]), 600)}")
            prompt = "\n".join(prompt_lines)

            with st.spinner("🧠 Analisando com GPT..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
            content = (resp.choices[0].message.content or "").strip()
            st.markdown("### 📄 Saída do Modelo (para auditoria)")
            st.text_area("Retorno do GPT", value=content, height=220)
            triples = _parse_dependencies_with_reasons(content)
        except Exception as e:
            st.error(f"❌ Falha ao consultar GPT: {e}")

    # ---------------- Fallback SBERT ----------------
    if not triples and use_fallback:
        st.warning("⚠️ Nenhuma ligação explícita do GPT. Aplicando fallback SBERT…")
        triples = _infer_semantic_links(subset, col_obj, n_top=2, thr=0.45)

    if not triples:
        st.error("❌ Nenhuma relação identificada (nem GPT nem SBERT).")
        export_zip_button(scope_key)
        return

    # ---------------- Gráfico ----------------
    st.markdown("### 🎨 Mapa de Dependências (A → B)")
    fig, levels = _draw_static_graph(triples, title="Mapa de Dependências entre UCs (A → B)")
    if fig is not None:
        show_and_export_fig(scope_key, fig, "grafo_dependencias_estatico")
        plt.close(fig)

    # ---------------- Tabelas (duas perspectivas) ----------------
    df_edges = pd.DataFrame(triples, columns=["UC (Pré-requisito)", "UC (Dependente)", "Justificativa"])
    # Camada (nível) por UC
    layer_df = pd.DataFrame(
        [{"UC": uc, "Nível (camada)": lv} for uc, lv in levels.items()]
    )

    st.markdown("### 📘 Visões em Tabela")
    tab1, tab2, tab3 = st.tabs(["Para cursar esta UC, preciso de…", "Esta UC prepara para…", "Todas as ligações (com justificativa)"])

    # 1) Por UC Destino: lista de pré-requisitos
    with tab1:
        req_by_dest = (
            df_edges.groupby("UC (Dependente)")
            .agg(**{
                "Pré-requisitos": ("UC (Pré-requisito)", lambda s: ", ".join(sorted(set(s)))),
                "Qtd Pré-requisitos": ("UC (Pré-requisito)", "nunique"),
            })
            .reset_index()
            .sort_values(["Qtd Pré-requisitos", "UC (Dependente)"], ascending=[False, True])
        )
        req_by_dest = req_by_dest.merge(layer_df, left_on="UC (Dependente)", right_on="UC", how="left").drop(columns=["UC"])
        st.dataframe(req_by_dest, use_container_width=True, hide_index=True)
        export_table(scope_key, req_by_dest, "dependencias_por_destino", "Pré-requisitos por UC (Destino)")

    # 2) Por UC Base: lista de dependentes
    with tab2:
        deps_by_base = (
            df_edges.groupby("UC (Pré-requisito)")
            .agg(**{
                "Dependentes": ("UC (Dependente)", lambda s: ", ".join(sorted(set(s)))),
                "Qtd Dependentes": ("UC (Dependente)", "nunique"),
            })
            .reset_index()
            .sort_values(["Qtd Dependentes", "UC (Pré-requisito)"], ascending=[False, True])
        )
        deps_by_base = deps_by_base.merge(layer_df, left_on="UC (Pré-requisito)", right_on="UC", how="left").drop(columns=["UC"])
        st.dataframe(deps_by_base, use_container_width=True, hide_index=True)
        export_table(scope_key, deps_by_base, "dependencias_por_base", "Dependentes por UC (Pré-requisito)")

    # 3) Edges detalhados
    with tab3:
        st.dataframe(df_edges, use_container_width=True, hide_index=True)
        export_table(scope_key, df_edges, "dependencias_edges", "Relações A → B com Justificativa")

    # ---------------- Análise interpretativa ----------------
    st.markdown("### 🧾 Análise Interpretativa dos Resultados")
    st.markdown(_analysis_text(triples))

    # ---------------- Métricas rápidas ----------------
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("UCs consideradas", len(subset))
    c2.metric("Relações A → B", len(df_edges))
    c3.metric("Camadas (níveis) no fluxo", len(set(levels.values())))

    # Export ZIP do escopo
    export_zip_button(scope_key)
