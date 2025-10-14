# ===============================================================
# 🧩 EmentaLabv2 — Text Helpers
# ===============================================================
import re
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List

def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower())

def find_col(df: pd.DataFrame, partial: str) -> Optional[str]:
    """Procura coluna por substring (case insensitive)"""
    for c in df.columns:
        if partial.lower() in c.lower():
            return c
    return None

def replace_semicolons(text: str) -> str:
    return re.sub(r"[;•\n]+", ". ", str(text))

def truncate(text: str, max_len: int = 250) -> str:
    """Trunca texto longo preservando palavras"""
    t = str(text)
    return t if len(t) <= max_len else t[:max_len].rsplit(" ", 1)[0] + "..."

def _split_sentences(text: str) -> List[str]:
    sents = re.split(r"[.!?;]\s+", str(text))
    return [s.strip() for s in sents if s.strip()]

def _rotate_xticks(ax, size=9):
    for tick in ax.get_xticklabels(): tick.set_rotation(45); tick.set_fontsize(size)

def _rotate_yticks(ax, size=9):
    for tick in ax.get_yticklabels(): tick.set_fontsize(size)
