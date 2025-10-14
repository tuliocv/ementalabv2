# ===============================================================
# 🔗 EmentaLabv2 — Grafo de Dependências (v11.1 — Direcional + Justificativas)
# ===============================================================
import re
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import streamlit as st

from utils.text_utils import find_col, truncate
from utils.embeddings import sbert_embed, l2_normalize
from utils.exportkit import export_table, export_zip_button


# ---------------------------------------------------------------
# 🔍 Extração de relações explícitas e justificativas
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


# ---------------------------------------------------------------
# 🤖 Fallback SBERT automático (quando GPT não é usado)
# ---------------------------------------------------------------
def _infer_semantic_links(df, col_text, n_top=2):
    """
    Gera pares A -> B quando há similaridade semântica alta entre conteúdos.
    """
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
            if sims[i, j] > 0.45:
                reason = f"Similaridade semântica de {sims[i, j]:.2f} entre conteúdos de {nome_a} e {nomes[j]}"
                triples.append((nome_a, nomes[j], reason))
    triples = list({(a, b, r) for a, b, r in triples if a != b})
    return triples


# ---------------------------------------------------------------
# 🎨 Desenho do grafo com setas direcionais e justificativas
# ---------------------------------------------------------------
def _draw_static_graph(pairs, show_labels=False):
    """
    Desenha o grafo com layout hierárquico da esquerda para a direita,
    com setas grandes e labels opcionais nas arestas.
    """
    if not pairs:
        return None

    G = nx.DiGraph()
    for a, b, reason in pairs:
        G.add_edge(a, b, label=reason)

    # Layout hierárquico (Graphviz se disponível, fallback = spring)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    except Exception:
        pos = nx.spring_layout(G, k=1.0, seed=42)

    plt.figure(figsize=(14, 8))
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="#a5d8ff", edgecolors="#1c7ed6", linewidths=1.5)

    # 🔹 Setas direcionais
    nx.draw_networkx_edges(
        G, pos,
        edge_color="#1c7ed6",
        width=2.2,
        alpha=0.9,
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.05",
    )

    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_color="#0b132b")

    # 🔹 Exibir justificativas sobre as arestas (opcional)
    if show_labels:
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_size=7, font_color="#2b2b2b", label_pos=0.55, rotate=False
        )

    plt.title("Mapa de Dependências entre UCs", fontsize=14, fontweight="bold", pad=20)
    plt.axis("off")
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()


# ---------------------------------------------------------------
# 🚀 Função principal (com client opcional)
# ---------------------------------------------------------------
def run_graph(df, scope_key, client=None):
    st.header("🔗 Dependência Curricular")
    st.caption(
        "Identifica relações de precedência e interdependência entre UCs com base em inferência semântica."
    )

    # -----------------------------------------------------------
    # 🧱 Identificação da coluna base
    # -----------------------------------------------------------
    col_obj = find_col(df, "Objetos de conhecimento") or find_col(df, "Conteúdo programático")
    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' (ou 'Conteúdo programático') não encontrada.")
        return

    subset = df[["Nome da UC", col_obj]].dropna()
    if subset.empty:
        st.warning("Nenhuma UC com 'Objetos de conhecimento' preenchido.")
        return

    max_uc = st.slider("Quantidade de UCs (amostra para análise)", 4, min(40, len(subset)), min(12, len(subset)), 1)
    subset = subset.head(max_uc)

    use_fallback = st.checkbox("⚙️ Ativar fallback automático SBERT", value=True)
    show_labels = st.checkbox("💬 Mostrar justificativas no grafo", value=False)

    # -----------------------------------------------------------
    # 🧠 Etapa 1 — Inferência GPT (se disponível)
    # -----------------------------------------------------------
    triples = []
    if client is not None:
        with st.spinner("🧠 Gerando análise via GPT..."):
            prompt_lines = [
                "Você deve identificar relações de pré-requisito entre as Unidades Curriculares (UCs) listadas.",
                "Responda somente no formato 'A -> B: justificativa'.",
                "",
                "Exemplo:",
                "Cálculo I -> Cálculo II: Cálculo I fornece as bases matemáticas para Cálculo II.",
                "",
                "UCs (nome: objetos de conhecimento):",
            ]
            for _, r in subset.iterrows():
                prompt_lines.append(f"- {r['Nome da UC']}: {truncate(str(r[col_obj]), 600)}")
            prompt = "\n".join(prompt_lines)

            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                content = (resp.choices[0].message.content or "").strip()
                triples = _parse_dependencies_with_reasons(content)
            except Exception as e:
                st.warning(f"❌ Falha na análise via GPT: {e}")

    # -----------------------------------------------------------
    # 🧩 Etapa 2 — Fallback SBERT
    # -----------------------------------------------------------
    if not triples and use_fallback:
        st.warning("⚠️ Nenhuma relação explícita detectada. Usando fallback SBERT…")
        triples = _infer_semantic_links(subset, col_obj)

    if not triples:
        st.error("❌ Nenhuma relação identificada (nem GPT nem SBERT).")
        export_zip_button(scope_key)
        return

    # -----------------------------------------------------------
    # 🎨 Etapa 3 — Visualização do Grafo
    # -----------------------------------------------------------
    st.markdown("### 🎨 Mapa de Dependências entre UCs")
    _draw_static_graph(triples, show_labels=show_labels)

    # -----------------------------------------------------------
    # 📊 Etapa 4 — Tabela de Relações
    # -----------------------------------------------------------
    df_edges = pd.DataFrame(triples, columns=["UC (Pré-requisito)", "UC Dependente", "Justificativa"])
    st.markdown("### 📘 Relações Identificadas e Justificativas")
    st.dataframe(df_edges, use_container_width=True, hide_index=True)
    export_table(scope_key, df_edges, "grafo_estatico_pre_requisitos", "Relações Pré-requisito")
    export_zip_button(scope_key)

    # -----------------------------------------------------------
    # 📈 Etapa 5 — Métricas
    # -----------------------------------------------------------
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("UCs analisadas", len(subset))
    c2.metric("Relações identificadas", len(triples))

    # -----------------------------------------------------------
    # 📘 Etapa 6 — Interpretação (exibida sempre)
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("📘 Como interpretar o gráfico")
    st.markdown(
        """
        ### 🔹 Leitura do Mapa
        - Cada **nó** representa uma Unidade Curricular (UC).
        - Cada **seta** indica uma **relação de dependência** (A → B = A é pré-requisito de B).
        - O grafo é desenhado da **esquerda para a direita**, representando o avanço formativo.
        - UCs mais à esquerda são **fundamentais**, enquanto as mais à direita **dependem de múltiplas bases**.

        ### 🔹 Análises Possíveis
        - **Coerência vertical** → verifica se as UCs seguem uma progressão lógica e cognitiva.
        - **Lacunas curriculares** → UCs isoladas ou desconectadas podem indicar falta de articulação.
        - **Densidade de conexões** → número alto de setas sugere integração interdisciplinar.

        ### 🔹 Aplicações Práticas
        - Validar **pré-requisitos pedagógicos** entre disciplinas.
        - Identificar **inconsistências de encadeamento** (UC avançada sem base clara).
        - Apoiar revisões de **matrizes curriculares**, fluxos de aprendizagem e PPCs.
        """
    )
