# ===============================================================
# 🎓 EmentaLabv2 — Bloom Heurístico
# ===============================================================
import re
import pandas as pd

BLOOM_VERBS = {
    "Lembrar": ["definir", "listar", "identificar", "reconhecer"],
    "Compreender": ["explicar", "descrever", "classificar", "resumir"],
    "Aplicar": ["usar", "executar", "demonstrar", "resolver"],
    "Analisar": ["comparar", "organizar", "diferenciar", "relacionar"],
    "Avaliar": ["julgar", "avaliar", "argumentar", "criticar"],
    "Criar": ["propor", "desenvolver", "planejar", "conceber"]
}

def detectar_bloom(texto):
    t = texto.lower()
    for nivel, verbos in BLOOM_VERBS.items():
        if any(v in t for v in verbos):
            return nivel
    return "Indefinido"

def calculate_bloom_level(df, col_obj):
    """Aplica heurística de Bloom em uma coluna"""
    out = []
    for _, r in df.iterrows():
        frase = str(r.get(col_obj, "")).strip()
        nivel = detectar_bloom(frase)
        out.append({
            "Nome da UC": r.get("Nome da UC", "—"),
            "Objetivo": frase,
            "Nível Bloom Predominante": nivel
        })
    return pd.DataFrame(out)
