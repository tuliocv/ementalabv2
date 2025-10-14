# ===============================================================
# 🧭 Matriz Integrada: Objetos × Egresso × DCN × Competências
# ===============================================================
def run_alignment_matrix(df: pd.DataFrame, scope_key: str, client=None):
    st.header("🧭 Matriz Integrada — Objetos × Egresso × DCN × Competências")
    st.caption(
        """
        Linhas representam as **Unidades Curriculares (UCs)**.  
        As colunas indicam o grau de similaridade semântica entre os **Objetos de Conhecimento**
        e as **Competências esperadas do Egresso**, **Relação DCN** e **Competências declaradas da UC**.
        
        Valores próximos de **1.00 (verde)** indicam **forte coerência curricular**,
        enquanto valores mais baixos (**vermelho**) indicam **possível desalinhamento**.
        """
    )

    col_obj = find_col(df, "Objetos de conhecimento")
    col_egresso = find_col(df, "Competências do Perfil do Egresso")
    col_dcn = find_col(df, "Competências DCN")
    col_comp = find_col(df, "Competências")
    if not col_obj:
        st.error("Coluna 'Objetos de conhecimento' não encontrada.")
        return
    if not (col_egresso or col_dcn or col_comp):
        st.error("Nenhuma coluna de competências encontrada (Egresso, DCN ou Competências).")
        return

    df_valid = df.fillna("")
    nomes = df_valid["Nome da UC"].astype(str).tolist()

    # Seleção do modo de cálculo
    modo = st.radio(
        "Selecione o tipo de cálculo de similaridade:",
        ["📊 Por Média da UC", "🧩 Frase a Frase"],
        horizontal=True,
        key="align_mode_radio_v2"
    )

    # ----------------------- MODO 1: Média por UC -----------------------
    if modo == "📊 Por Média da UC":
        textos_obj = df_valid[col_obj].astype(str).apply(replace_semicolons).tolist()
        emb_obj = l2_normalize(sbert_embed(textos_obj))

        results = {"UC": nomes}
        if col_egresso:
            emb_egr = l2_normalize(sbert_embed(df_valid[col_egresso].astype(str).tolist()))
            results["Similaridade Perfil do Egresso"] = np.diag(np.dot(emb_obj, emb_egr.T))
        if col_dcn:
            emb_dcn = l2_normalize(sbert_embed(df_valid[col_dcn].astype(str).tolist()))
            results["Similaridade Relação DCN"] = np.diag(np.dot(emb_obj, emb_dcn.T))
        if col_comp:
            emb_comp = l2_normalize(sbert_embed(df_valid[col_comp].astype(str).tolist()))
            results["Similaridade Competências"] = np.diag(np.dot(emb_obj, emb_comp.T))

        df_res = pd.DataFrame(results)

    # ----------------------- MODO 2: Frase a Frase -----------------------
    else:
        rows = []
        for _, row in df_valid.iterrows():
            nome = str(row["Nome da UC"])
            obj_text = replace_semicolons(str(row.get(col_obj, "")))
            egr_text = replace_semicolons(str(row.get(col_egresso, "")))
            dcn_text = replace_semicolons(str(row.get(col_dcn, "")))
            comp_text = replace_semicolons(str(row.get(col_comp, "")))

            emb_obj = sbert_embed(_split_sentences(obj_text))
            sim_egr = sim_dcn = sim_comp = np.nan

            if egr_text.strip():
                emb_egr = sbert_embed(_split_sentences(egr_text))
                sim_egr = float(np.mean(cosine_similarity(emb_obj, emb_egr)))
            if dcn_text.strip():
                emb_dcn = sbert_embed(_split_sentences(dcn_text))
                sim_dcn = float(np.mean(cosine_similarity(emb_obj, emb_dcn)))
            if comp_text.strip():
                emb_comp = sbert_embed(_split_sentences(comp_text))
                sim_comp = float(np.mean(cosine_similarity(emb_obj, emb_comp)))

            rows.append({
                "UC": nome,
                "Similaridade Perfil do Egresso": sim_egr,
                "Similaridade Relação DCN": sim_dcn,
                "Similaridade Competências": sim_comp
            })

        df_res = pd.DataFrame(rows)

    # ----------------------- Exibição -----------------------
    st.markdown("### 📊 Matriz Integrada de Similaridade")
    st.dataframe(safe_style(df_res, cmap="RdYlGn"), use_container_width=True)
    export_table(scope_key, df_res, "matriz_alinhamento_completa", "Matriz Integrada Objetos × Egresso × DCN × Competências")

    # Heatmap
    fig, ax = plt.subplots(figsize=(9, 5))
    df_plot = df_res.set_index("UC")
    for c in df_plot.columns:
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
    sns.heatmap(df_plot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Mapa de Similaridade Curricular Integrado", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, use_container_width=True)

    # ----------------------- Relatório GPT -----------------------
    if client is None:
        api_key = st.session_state.get("global_api_key", "")
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
            except Exception:
                client = None

    st.markdown("---")
    st.subheader("🧾 Relatório Analítico de Alinhamento Curricular")

    if client:
        medias = df_res.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").mean()
        ucs_baixas = df_res[df_res.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").mean(axis=1) < 0.65]["UC"].tolist()
        resumo = {
            "modo": modo,
            "media_egresso": float(medias.get("Similaridade Perfil do Egresso", np.nan)),
            "media_dcn": float(medias.get("Similaridade Relação DCN", np.nan)),
            "media_comp": float(medias.get("Similaridade Competências", np.nan)),
            "ucs_baixas": ucs_baixas,
        }

        prompt = f"""
        Você é um avaliador curricular. Analise os seguintes dados:
        {resumo}

        Gere um relatório técnico breve (até 150 palavras) com:
        - Pontos fortes
        - Fragilidades
        - Recomendações práticas
        Linguagem objetiva, voltada à melhoria curricular.
        """

        try:
            with st.spinner("🧠 Gerando relatório via GPT..."):
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            analise = resp.choices[0].message.content.strip()
            st.success("Relatório gerado com sucesso.")
            st.markdown(analise)
        except Exception as e:
            st.error(f"Erro ao gerar relatório via GPT: {e}")
    else:
        st.info("🔑 Chave da OpenAI não encontrada — relatório não gerado.")

    export_zip_button(scope_key)

    # ----------------------- Interpretação -----------------------
    st.markdown("---")
    st.subheader("📘 Como interpretar esta matriz")
    st.markdown(
        """
        - **Verde (≥ 0.85):** Forte coerência semântica entre objetivos e competências.  
        - **Amarelo (0.65–0.85):** Coerência moderada, possível necessidade de reescrita.  
        - **Vermelho (< 0.65):** Fraco alinhamento, revisar conteúdos ou competências.  

        **Modo “Frase a Frase”**: detecta desalinhamentos textuais sutis.  
        **Modo “Por Média da UC”**: visão geral da coerência pedagógica global.
        """
    )
