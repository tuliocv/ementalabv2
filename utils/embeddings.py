# ===============================================================
# 🔬 EmentaLabv2 — Embeddings SBERT
# ===============================================================
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner=False)
def sbert_model():
    """Carrega modelo SBERT com cache para performance."""
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def sbert_embed(texts):
    """Gera embeddings vetoriais normalizados"""
    model = sbert_model()
    return np.array(model.encode(texts, show_progress_bar=False))

def l2_normalize(mat):
    """Normaliza vetores linha a linha"""
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)
