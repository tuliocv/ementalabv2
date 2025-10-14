# ===============================================================
# 🔬 EmentaLabv2 — Embeddings SBERT
# ===============================================================
import numpy as np
from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner=False)
def sbert_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def sbert_embed(texts):
    """Gera embeddings vetoriais normalizados"""
    model = sbert_model()
    return np.array(model.encode(texts, show_progress_bar=False))

def l2_normalize(mat):
    """Normaliza vetores linha a linha"""
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)
