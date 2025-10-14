import unicodedata

def normalize_text(txt):
    if not isinstance(txt, str):
        return ""
    txt = unicodedata.normalize("NFKD", txt).encode("ASCII", "ignore").decode("utf-8")
    return txt.lower().strip()
