# ===============================================================
# 🤖 EmentaLabv2 — Relatório Consultivo (v3.0)
# ===============================================================
# - Assinatura compatível com app.py: run_consultive(df, scope_key, client=None)
# - Gera diagnóstico rápido (sem GPT) + relatório curto (com GPT se API estiver disponível)
# - Usa a API key global do sidebar via st.session_state["global_api_key"], caso client não seja passado
# - Exporta tabelas/figuras e inclui tudo no ZIP do escopo
# ===============================================================

from __future__ import annotations
import json
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# GPT opcional
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from utils.text_utils import find_col
from utils.exportkit import export_table, show_and_export_fig, export_zip_button, _init_exports


# ------------------------ Helpers ------------------------

def _safe_len_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.len()

def _snapshot_basico(df: pd.DataFrame) -> dict:
    """
    Coleta indicadores simples e estáveis para alimentar o relatório:
    - contagem de UCs, cursos etc.
    - presença de colunas-chave
    - métricas textuais básicas (tamanho de ementa/objetivo)
    """
    col_uc   = find_col(df, "Nome da UC")
    col_cur  = find_col(df, "Nome do curso")
    col_em   = find_col(df, "Ementa") or find_col(df, "Descrição")
    col_obj  = find_col(df, "Objetivo de aprendizagem")
    col_comp = find_col(df, "Competências do Perfil do Egresso")

    snap = {
        "n_rows": int(len(df)),
        "tem_col_uc": bool(col_uc),
        "tem_col_curso": bool(col_cur),
        "tem_col_ementa": bool(col_em),
        "tem_col_obj": bool(col_obj),
        "tem_col_comp": bool(col_comp),
        "col_uc": col_uc,
        "col_curso": col_cur,
        "col_ementa": col_em,
        "col_obj": col_obj,
        "col_comp": col_comp,
    }

    if col_uc:
        snap["n_ucs"] = int(df[col_uc].dropna().nunique())
    else:
        snap["n_ucs"] = int(df.dropna().shape[0])

    if col_cur:
        snap["n_cursos"] = int(df[col_cur].dropna().nunique())
    else:
        snap["n_cursos"] = 0

    if col_em:
        lens_em = _safe_len_series(df[col_em])
        snap["ementa_len_avg"] = float(lens_em.mean())
        snap["ementa_len_med"] = float(lens_em.median())
        snap["ementa_pct_vazia"] = float((lens_em == 0).mean()*100)
    else:
        snap["ementa_len_avg"] = snap["ementa_len_med"] = 0.0
        snap["ementa_pct_vazia"] = 0.0

    if col_obj:
        lens_obj = _safe_len_series(df[col_obj])
        snap["obj_len_avg"] = float(lens_obj.mean())
        snap["obj_len_med"] = float(lens_obj.median())
        snap["obj_pct_vazio"] = float((lens_obj == 0).mean()*100)
    else:
        snap["obj_len_avg"] = snap["obj_len_med"] = 0.0
        snap["obj_pct_vazio"] = 0.0

    return snap


def _plot_len_distribution(scope_key: str, series: pd.Series, title: str, base_name: str):
    """Desenha histograma simples do comprimento textual."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(series, bins=20, kde=True, ax=ax, color="#3b5bdb")
    ax.set_title(title)
    ax.set_xlabel("Número de caracteres")
    ax.set_ylabel("Quantidade de UCs")
    show_and_export_fig(scope_key, fig, base_name)
    plt.close(fig)


def _to_bullets(d: dict) -> str:
    """Converte um dicionário em lista de bullets legível."""
    lines = []
    for k, v in d.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


# ------------------------ Módulo principal ------------------------

def run_consultive(df: pd.DataFrame, scope_key: str, client=None):
    st.header("🤖 Relatório Consultivo")
    st.caption(
        "Gera um diagnóstico objetivo da base carregada e, se disponível uma chave da OpenAI, "
        "complementa com um **parecer técnico curto (pontos fortes, fragilidades e recomendações práticas)**."
    )

    if df is None or df.empty:
        st.error("Base vazia. Carregue um arquivo para continuar.")
        return

    # snapshot
    snap = _snapshot_basico(df)

    # seção: métricas gerais
    st.subheader("📊 Visão Geral")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros (linhas)", f"{snap['n_rows']}")
    c2.metric("UCs únicas", f"{snap['n_ucs']}")
    c3.metric("Cursos únicos", f"{snap['n_cursos']}")
    c4.metric("Objetivos vazios", f"{snap['obj_pct_vazio']:.1f}%")

    cols_presentes = {
        "Nome da UC":        "✅" if snap["tem_col_uc"] else "❌",
        "Nome do curso":     "✅" if snap["tem_col_curso"] else "❌",
        "Ementa/Descrição":  "✅" if snap["tem_col_ementa"] else "❌",
        "Objetivo de apr.":  "✅" if snap["tem_col_obj"] else "❌",
        "Competências":      "✅" if snap["tem_col_comp"] else "❌",
    }
    st.markdown("### 🧾 Colunas essenciais")
    st.markdown(_to_bullets(cols_presentes))

    # tabelas de apoio
    tables_to_export = {}

    # distribuição de tamanhos (ementa/objetivos)
    if snap["tem_col_ementa"]:
        em_series = _safe_len_series(df[snap["col_ementa"]])
        st.markdown("### 🗂️ Tamanho das Ementas (caracteres)")
        _plot_len_distribution(scope_key, em_series, "Distribuição do tamanho das ementas", "consultivo_dist_ementa")
        tables_to_export["tamanho_ementa"] = pd.DataFrame({"tamanho_ementa": em_series})

    if snap["tem_col_obj"]:
        obj_series = _safe_len_series(df[snap["col_obj"]])
        st.markdown("### 🗂️ Tamanho dos Objetivos de Aprendizagem (caracteres)")
        _plot_len_distribution(scope_key, obj_series, "Distribuição do tamanho dos objetivos", "consultivo_dist_objetivos")
        tables_to_export["tamanho_objetivos"] = pd.DataFrame({"tamanho_objetivos": obj_series})

    # exporta tabelas auxiliares
    for name, tdf in tables_to_export.items():
        export_table(scope_key, tdf, f"consultivo_{name}", f"Consultivo — {name}")

    # snapshot em tabela para export
    df_snapshot = pd.DataFrame(
        [
            {"Indicador": "Linhas (registros)", "Valor": snap["n_rows"]},
            {"Indicador": "UCs únicas", "Valor": snap["n_ucs"]},
            {"Indicador": "Cursos únicos", "Valor": snap["n_cursos"]},
            {"Indicador": "% objetivos vazios", "Valor": f"{snap['obj_pct_vazio']:.1f}%"},
            {"Indicador": "Tamanho médio ementa", "Valor": f"{snap['ementa_len_avg']:.0f}"},
            {"Indicador": "Tamanho médio objetivos", "Valor": f"{snap['obj_len_avg']:.0f}"},
        ]
    )
    st.markdown("### 📋 Indicadores (snapshot)")
    st.dataframe(df_snapshot, hide_index=True, use_container_width=True)
    export_table(scope_key, df_snapshot, "consultivo_snapshot", "Consultivo — Snapshot")

    # ------------- GPT (opcional) -------------
    # Prioriza client passado pelo app; senão tenta pegar do session_state
    if client is None and OpenAI is not None:
        api_key = st.session_state.get("global_api_key", "")
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
            except Exception:
                client = None

    st.markdown("---")
    st.subheader("🧾 Parecer Técnico (automático)")

    if client is None:
        # Relatório objetivo local (sem GPT)
        st.info(
            "Chave da OpenAI não encontrada. Abaixo um parecer **objetivo e direto** gerado localmente:"
        )
        parecer_local = _parecer_local(snap)
        st.markdown(parecer_local)
        # exporta como .txt
        _export_text(scope_key, "consultivo_parecer_local", parecer_local)
        export_zip_button(scope_key)
        return

    # Monta resumo objetivo para o prompt
    resumo = {
        "ucs_unicas": snap["n_ucs"],
        "cursos_unicos": snap["n_cursos"],
        "pct_objetivos_vazios": round(snap["obj_pct_vazio"], 1),
        "tam_medio_ementa": round(snap["ementa_len_avg"], 0),
        "tam_medio_objetivo": round(snap["obj_len_avg"], 0),
        "colunas_presentes": {
            "UC": snap["tem_col_uc"],
            "Curso": snap["tem_col_curso"],
            "Ementa/Descrição": snap["tem_col_ementa"],
            "Objetivo": snap["tem_col_obj"],
            "Competências": snap["tem_col_comp"],
        },
    }

    prompt = textwrap.dedent(f"""
    Você é um avaliador curricular. Com base no resumo (JSON) a seguir,
    produza um **parecer técnico curto e direto**, com 3 seções:
    1) **Pontos fortes** (3 bullets)
    2) **Fragilidades** (3 bullets)
    3) **Recomendações práticas priorizadas** (até 5 bullets, cada bullet no formato: ação + impacto esperado)

    Regras:
    - Linguagem objetiva, sem adjetivos excessivos
    - Sem repetir números do resumo; foque em implicações e ações
    - Evite jargões; mantenha foco pedagógico-operacional
    - Máximo de 160 palavras

    JSON de entrada:
    {json.dumps(resumo, ensure_ascii=False)}
    """)

    try:
        with st.spinner("Gerando parecer técnico (GPT)…"):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
        parecer_gpt = (resp.choices[0].message.content or "").strip()
        st.success("Parecer gerado com sucesso.")
        st.markdown(parecer_gpt)
        _export_text(scope_key, "consultivo_parecer_gpt", parecer_gpt)
    except Exception as e:
        st.error(f"Não foi possível gerar o parecer com GPT: {e}")
        parecer_local = _parecer_local(snap)
        st.markdown(parecer_local)
        _export_text(scope_key, "consultivo_parecer_local_fallback", parecer_local)

    export_zip_button(scope_key)


# ------------------------ Utilidades internas ------------------------

def _parecer_local(snap: dict) -> str:
    """Parecer curto sem GPT — objetivo e direto."""
    pontos_fortes = []
    fragilidades = []
    recomendacoes = []

    # forças
    if snap["tem_col_ementa"]:
        pontos_fortes.append("Ementas presentes em boa parte das UCs, viabilizando leitura temática.")
    if snap["tem_col_obj"]:
        pontos_fortes.append("Campo de objetivos disponível para análise de clareza e Bloom.")
    if snap["tem_col_comp"]:
        pontos_fortes.append("Competências mapeadas, permitindo avaliar coerência com objetivos.")

    # fragilidades
    if snap["obj_pct_vazio"] > 10:
        fragilidades.append("Percentual relevante de UCs com objetivos ausentes ou vazios.")
    if snap["ementa_len_avg"] < 200:
        fragilidades.append("Ementas curtas: risco de escopo pouco claro.")
    if not snap["tem_col_comp"]:
        fragilidades.append("Ausência de competências reduz o alinhamento com o perfil do egresso.")

    if not fragilidades:
        fragilidades.append("Não foram detectadas fragilidades estruturais a partir dos campos básicos.")

    # recomendações (priorizadas)
    if snap["obj_pct_vazio"] > 10:
        recomendacoes.append("Revisar e completar objetivos ausentes (padronizar verbos-ação).")
    if snap["ementa_len_avg"] < 200:
        recomendacoes.append("Ampliar ementas com tópicos-chave e resultados esperados.")
    if not snap["tem_col_comp"]:
        recomendacoes.append("Incluir competências por UC para viabilizar análises de coerência.")
    recomendacoes.append("Utilizar os módulos Bloom e Alinhamento para ajustes finos de coerência.")
    recomendacoes.append("Gerar mapa de dependências para validar sequência de pré-requisitos.")

    def bullets(xs): 
        return "\n".join([f"- {x}" for x in xs])

    txt = f"""
**Pontos fortes**
{bullets(pontos_fortes)}

**Fragilidades**
{bullets(fragilidades)}

**Recomendações práticas (priorizadas)**
{bullets(recomendacoes[:5])}
""".strip()
    return txt


def _export_text(scope_key: str, filename: str, content: str):
    """Salva um .txt no diretório do escopo para baixar no ZIP."""
    export_dir = _init_exports(scope_key)
    path = f"{export_dir}/{filename}.txt"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        st.error(f"Erro ao salvar relatório de texto: {e}")
