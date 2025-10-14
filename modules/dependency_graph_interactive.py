# ===============================================================
# 🔗 EmentaLabv2 — Grafo Interativo de Dependências (v8.5)
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

# ---------------------------------------------------------------
# 🔍 Funções auxiliares
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
    """Cria pares prováveis com base em similaridade SBERT (fallback automático)."""
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


# ---------------------------------------------------------------
# 🚀 Função principal
# ---------------------------------------------------------------
def run_graph_interactive(df: pd.DataFrame, scope_key: str):
    st.header("🌐 Grafo Interativo — Relações de Pré-requisito entre UCs")
    st.markdown("""
    Este módulo identifica **relações de dependência e precedência** entre as **Unidades Curriculares (UCs)**.
    A análise permite **visualizar como o conhecimento se encadeia** ao longo da matriz curricular — 
    revelando **sequências de aprendizagem**, **interdependências** e **lacunas estruturais**.

    ---
    **Como funciona:**
    - O modelo GPT lê os **objetos de conhecimento** (ou conteúdos programáticos) de cada UC e infere relações do tipo “A → B” (A é pré-requisito de B).
    - Quando o GPT não encontra relações explícitas, o sistema utiliza **SBERT** como fallback semântico, identificando pares de UCs com alto grau de similaridade conceitual.
    - O resultado é exibido como um **grafo interativo**, onde **nós** representam UCs e **setas** indicam precedência de aprendizagem.
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

    # ---------------- Execução GPT ----------------
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

    # ---------------- Gera o grafo ----------------
    html_path = _draw_interactive_graph(pairs)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=700, scrolling=True)

    # ---------------- Exportações ----------------
    df_edges = pd.DataFrame(pairs, columns=["Pré-requisito", "UC Dependente"])
    export_table(scope_key, df_edges, "grafo_interativo_pre_requisitos", "Relações Pré-requisito (Interativo)")
    export_zip_button(scope_key)

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("UCs analisadas", len(subset))
    c2.metric("Relações identificadas", len(pairs))

    # -----------------------------------------------------------
    # 🧠 Interpretação e leitura pedagógica (explicativa)
    # -----------------------------------------------------------
    with st.expander("🧭 Como interpretar o grafo e aplicar os resultados", expanded=False):
        st.markdown("""
        ### 🔹 1. O que o grafo mostra

        Cada **nó** representa uma Unidade Curricular (UC) e cada **seta** indica uma **relação de precedência**:
        - **A → B** significa que os conteúdos ou competências da UC **A** são pré-requisitos para compreender **B**.  
        - UCs com **muitas conexões de saída** (várias setas partindo delas) indicam **fundamentos** ou **disciplinas-base**.  
        - UCs com **muitas conexões de entrada** indicam **disciplinas de síntese**, que dependem de vários conhecimentos anteriores.

        ---

        ### 🔹 2. Como interpretar a estrutura

        | Tipo de nó | Interpretação pedagógica | Exemplo típico |
        |-------------|---------------------------|----------------|
        | **Nó central com muitas saídas** | Fundamento formativo, base conceitual | Matemática, Programação I |
        | **Nó periférico isolado** | UC independente ou de eixo transversal | Ética, Empreendedorismo |
        | **Nó com muitas entradas** | Integração de saberes (síntese) | Projeto Integrador, Trabalho de Conclusão |
        | **Cadeia linear (A → B → C)** | Sequência progressiva de aprendizagem | Física I → Física II → Termodinâmica |

        ---

        ### 🔹 3. Finalidade da análise

        - **Verificar coerência curricular:** se as UCs avançadas dependem de bases sólidas e corretamente ordenadas.  
        - **Detectar lacunas:** se há UCs que não possuem nenhuma conexão de entrada ou saída, o que pode indicar
          ausência de integração ou conteúdos isolados.  
        - **Evidenciar sobreposições:** se várias UCs compartilham dependências semelhantes, pode haver redundância.  
        - **Apoiar revisões de PPC e NDE:** o grafo funciona como um “mapa de coerência” do fluxo de aprendizagem.

        ---

        ### 🔹 4. Dicas de leitura

        - **Navegue e amplie o grafo**: arraste os nós e observe agrupamentos automáticos por área temática.  
        - **Clique sobre os nós** para destacar suas dependências diretas.  
        - **Observe o sentido das setas**: o fluxo ideal vai das bases para as sínteses (da esquerda para a direita).  
        - **Use o número de relações** (métricas abaixo) para dimensionar a densidade de conexões.

        ---

        ### 🔹 5. Exemplo de interpretação prática

        ```
        Expressão e Linguagens Visuais → Meios de Representação
        Meios de Representação → Projeto de Interiores Residenciais
        Projeto de Interiores Residenciais → Projeto de Habitação Unifamiliar
        ```

        ✳️ **Leitura:**  
        - Mostra uma **cadeia formativa progressiva**: primeiro a base de expressão visual, depois representação técnica e, por fim, aplicação em projetos.  
        - Isso reflete **uma progressão cognitiva de complexidade**, coerente com o desenvolvimento de competências profissionais.

        ---

        ### 🔹 6. Conclusão

        Este grafo evidencia o **encadeamento lógico-pedagógico** da matriz curricular, 
        revelando **como o conhecimento se propaga entre UCs**.  
        Ele é uma ferramenta estratégica para:
        - Revisar coerência vertical da matriz;  
        - Apoiar revisões do PPC;  
        - Embasar relatórios de NDE e autoavaliação institucional (CPA).

        """)
