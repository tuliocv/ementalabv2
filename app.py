# ===============================================================
# 2. FILTROS DE CONTEXTO — EMENTALABv2
# ===============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filtros de Contexto")

# 🔹 Filtros individuais
if "Nome do curso" in df.columns:
    cursos = sorted(df["Nome do curso"].dropna().unique().tolist())
    curso_sel = st.sidebar.multiselect("📘 Nome do Curso", cursos)
    if curso_sel:
        df = df[df["Nome do curso"].isin(curso_sel)]

if "Modalidade do curso" in df.columns:
    modalidades = sorted(df["Modalidade do curso"].dropna().unique().tolist())
    mod_sel = st.sidebar.multiselect("🏫 Modalidade do Curso", modalidades)
    if mod_sel:
        df = df[df["Modalidade do curso"].isin(mod_sel)]

if "Tipo Graduação" in df.columns:
    tipos = sorted(df["Tipo Graduação"].dropna().unique().tolist())
    tipo_sel = st.sidebar.multiselect("🎓 Tipo de Graduação", tipos)
    if tipo_sel:
        df = df[df["Tipo Graduação"].isin(tipo_sel)]

if "Cluster" in df.columns:
    clusters = sorted(df["Cluster"].dropna().unique().tolist())
    cluster_sel = st.sidebar.multiselect("🌐 Cluster", clusters)
    if cluster_sel:
        df = df[df["Cluster"].isin(cluster_sel)]

if "Tipo do componente" in df.columns:
    tipos_comp = sorted(df["Tipo do componente"].dropna().unique().tolist())
    tipo_comp_sel = st.sidebar.multiselect("🧩 Tipo do Componente", tipos_comp)
    if tipo_comp_sel:
        df = df[df["Tipo do componente"].isin(tipo_comp_sel)]

# 🔹 Exibir quantidade de registros após filtros
st.sidebar.markdown("---")
st.sidebar.info(f"📊 {len(df)} registros após aplicação dos filtros.")
