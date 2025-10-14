import pandas as pd
import streamlit as st
from textblob import TextBlob
import re

def sentiment_clarity(df, col_text="EMENTA"):
    st.header("✍️ Análise de Clareza e Sentimento das Ementas")

    def clarity_index(text):
        sents = re.split(r'[.!?]', text)
        words = text.split()
        return len(words) / max(1, len(sents))

    df["Clareza"] = df[col_text].apply(clarity_index)
    df["Sentimento"] = df[col_text].apply(lambda t: TextBlob(t).sentiment.polarity)

    st.metric("📏 Clareza Média", round(df["Clareza"].mean(), 2))
    st.metric("💬 Sentimento Médio", round(df["Sentimento"].mean(), 2))
    st.dataframe(df[[col_text, "Clareza", "Sentimento"]], use_container_width=True)
    st.success("Análise de clareza e sentimento concluída.")
    return df
