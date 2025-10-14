# ===============================================================
# 🔗 EmentaLabv2 — Grafo de Dependências (v11.7 — Relatório Analítico Objetivo)
# ===============================================================
# - Prompt GPT mais robusto e reprodutível
# - Relatório analítico objetivo e direto (pontos fortes, fracos e ações)
# - Mantém “Como interpretar o gráfico” sempre visível
# - Mostra UCs dependentes e antecessores
# - Compatível com o app principal
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
    """Extrai pares 'A -> B' e justificativas (quando houver)."""
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
    """Fallback SBERT para casos sem GPT."""
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
# 🎨 Desenho do grafo
# ---------------------------------------------------------------
def _draw_static_graph(pairs):
    """Desenha o grafo hierárquico (esquerda → direita)."""
    if not pairs:
        return None

    G = nx.DiGraph()
    for a, b, _ in pairs:
        G.add_edge(a, b)

    try:
        pos = nx.multipartite_layout(
            G,
            subset_key=lambda n: nx.shortest_path_length(G, list(G.nodes)[0], n)
            if nx.has_path(G, list(G.nodes)[0], n) else 0
        )
    except Exception:
        pos = nx.spring_layout(G, k=0.5, seed=42)

    plt.figure(figsize=(13, 8))
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="#cfe2ff", edgecolors="#084298")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=18, edge_color="#0d6efd", width=2, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=6, font_weight="bold", font_color="#1c1c1c")
    plt.title("Mapa de Dependências entre UCs", fontsize=14, fontweight="bold", pad=20)
    plt.axis("off")
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()


# ---------------------------------------------------------------
# 🚀 Função principal
# ---------------------------------------------------------------
def run_graph(df, scope_key, client=None):
    st.header("🔗 Dependência Curricular")
    st.caption(
        "Identifica relações de **precedência e interdependência** entre as Unidades Curriculares (UCs), "
        "a partir dos **objetos de conhecimento** ou **conteúdos programáticos**. "
        "Cada seta indica uma relação de **pré-requisito** (A → B)."
    )

    col_obj = find_col(df, "Objetos de conhecimento") or find_col(df, "Conteúdo programático")
    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' (ou 'Conteúdo programático') não encontrada.")
        return

    subset = df[["Nome da UC", col_obj]].dropna()
    if subset.empty:
        st.warning("Nenhuma UC com 'Objetos de conhecimento' preenchido.")
        return

    max_uc = st.slider("Quantidade de UCs (amostra para análise)", 4, min(40, len(subset)), min(14, len(subset)), 1)
    subset = subset.head(max_uc)

    use_fallback = st.checkbox("⚙️ Ativar fallback automático SBERT", value=True)
    api_key = st.session_state.get("global_api_key", "")
    if api_key:
        client = OpenAI(api_key=api_key)

    triples = []

    # -----------------------------------------------------------
    # 🧠 Etapa 1 — Inferência GPT (prompt robusto)
    # -----------------------------------------------------------
    if client is not None:
        with st.spinner("🧠 Gerando análise via GPT..."):
            prompt_lines = [
                "TAREFA: Identifique relações de PRÉ-REQUISITO (A -> B) entre as Unidades Curriculares listadas.",
                "DEFINIÇÃO: A é pré-requisito de B quando o conteúdo de A é necessário para cursar B.",
                "FORMATO ESTRITO: A -> B: justificativa curta",
                "REGRAS:",
                "- Use apenas o formato acima, sem texto adicional.",
                "- Evite repetições e relações triviais.",
                "",
                "EXEMPLO:",
                "Fundamentos de Cálculo -> Cálculo I: fornece base conceitual para derivadas.",
                "Cálculo I -> Cálculo II: base para integração e funções compostas.",
                "",
                "LISTA DE UCS:",
            ]
            for _, r in subset.iterrows():
                prompt_lines.append(f"- {r['Nome da UC']}: {truncate(str(r[col_obj]), 600)}")

            prompt = "\n".join(prompt_lines)

            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = (resp.choices[0].message.content or "").strip()
                st.markdown("### 📄 Saída do Modelo (para auditoria)")
                st.text_area("Retorno do GPT", value=content, height=220)
                triples = _parse_dependencies_with_reasons(content)

                if not triples:
                    st.warning("⚠️ GPT não retornou pares válidos, tentando regex…")
                    pattern = re.findall(r"([A-ZÁÉÍÓÚÂÊÔÃÕÇa-z0-9 ,\-()]+)\s*[-–>]{1,2}\s*([A-ZÁÉÍÓÚÂÊÔÃÕÇa-z0-9 ,\-()]+)", content)
                    if pattern:
                        triples = [(a.strip(), b.strip(), "Inferido do texto livre") for a, b in pattern]
                        st.info("⚙️ Relações inferidas automaticamente.")
            except Exception as e:
                st.warning(f"❌ Falha na análise via GPT: {e}")

    # -----------------------------------------------------------
    # 🧩 Etapa 2 — Fallback SBERT
    # -----------------------------------------------------------
    if not triples and use_fallback:
        st.warning("💡 Nenhuma relação explícita via GPT. Aplicando fallback SBERT…")
        triples = _infer_semantic_links(subset, col_obj)

    if not triples:
        st.error("❌ Nenhuma relação identificada (nem GPT nem SBERT).")
        export_zip_button(scope_key)
        return

    # -----------------------------------------------------------
    # 🎨 Etapa 3 — Visualização do Grafo
    # -----------------------------------------------------------
    st.markdown("### 🎨 Mapa de Dependências entre UCs (A → B)")
    _draw_static_graph(triples)

    # -----------------------------------------------------------
    # 📊 Etapa 4 — Tabelas explicativas
    # -----------------------------------------------------------
    df_edges = pd.DataFrame(triples, columns=["UC (Pré-requisito)", "UC (Dependente)", "Justificativa"])
    st.markdown("### 📘 Relações Identificadas")
    st.write(
        """
        - **UC (Pré-requisito):** Unidade Curricular que fornece base para outra.
        - **UC (Dependente):** Unidade que exige o conhecimento prévio da anterior.
        """
    )
    st.dataframe(df_edges, use_container_width=True, hide_index=True)
    export_table(scope_key, df_edges, "grafo_dependencias", "Relações de Dependência")

    # -----------------------------------------------------------
    # 🔁 Etapa 5 — Antecessores por UC
    # -----------------------------------------------------------
    st.markdown("### 🧩 Dependências por UC (Antecessores)")
    dependencias = (
        df_edges.groupby("UC (Dependente)")["UC (Pré-requisito)"]
        .apply(lambda x: ", ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"UC (Dependente)": "UC", "UC (Pré-requisito)": "Depende de"})
    )
    st.dataframe(dependencias, use_container_width=True, hide_index=True)
    export_table(scope_key, dependencias, "ucs_antecessores", "UCs e seus Pré-requisitos")

    # -----------------------------------------------------------
    # 📈 Etapa 6 — Métricas
    # -----------------------------------------------------------
    st.markdown("### 📈 Métricas Gerais")
    c1, c2 = st.columns(2)
    c1.metric("UCs analisadas", len(subset))
    c2.metric("Relações identificadas", len(triples))

    # -----------------------------------------------------------
    # 🧾 Etapa 8 — Relatório Analítico Objetivo
    # -----------------------------------------------------------
    if client is not None:
        with st.spinner("📝 Gerando relatório analítico da estrutura curricular..."):
            try:
                resumo_prompt = (
                    "Com base nas seguintes relações de dependência entre UCs:\n\n"
                    + "\n".join([f"{a} -> {b}: {r}" for a, b, r in triples[:60]]) +
                    "\n\nElabore um **relatório analítico objetivo** com no máximo 200 palavras, estruturado assim:\n\n"
                    "### Pontos Fortes\n"
                    "- (2 a 4 itens curtos e diretos)\n"
                    "### Pontos Fracos\n"
                    "- (2 a 4 itens curtos e diretos)\n"
                    "### Ações Recomendadas\n"
                    "- (2 a 4 recomendações práticas e assertivas para melhoria da coerência curricular)\n"
                    "Evite redundâncias e linguagem genérica."
                )
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": resumo_prompt}],
                    temperature=0.2,
                )
                analise_texto = (resp.choices[0].message.content or "").strip()
                st.markdown("### 🧾 Relatório Analítico da Estrutura Curricular")
                st.markdown(analise_texto)
            except Exception as e:
                st.warning(f"Não foi possível gerar o relatório analítico: {e}")

    # -----------------------------------------------------------
    # 🧭 Etapa 7 — Interpretação
    # -----------------------------------------------------------
    st.markdown("---")
    st.markdown(
        """
        ## 🧭 Como interpretar o gráfico
        - Cada **nó** representa uma Unidade Curricular (UC).
        - Cada **seta** indica uma **relação de dependência** (A → B = A é pré-requisito de B).
        - UCs à esquerda são **bases estruturantes**; UCs à direita dependem dessas fundações.
        - UCs isoladas podem representar **disciplinas autônomas ou desconectadas**.
        """
    )


    export_zip_button(scope_key)
