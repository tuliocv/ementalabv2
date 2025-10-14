from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def embed_texts(texts):
    model = load_sbert_model()
    return model.encode(texts, convert_to_tensor=True)
