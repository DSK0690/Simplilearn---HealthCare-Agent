"""Lightweight, dependency-free text embeddings (signed feature hashing).

Deterministic and works offline. If `sentence-transformers` is installed and EMBEDDING_BACKEND=sbert is set,
that model is used instead for higher-quality semantic matching.
"""
import hashlib
import os
import re

import numpy as np

from . import config

STOPWORDS = set("""a an and are as at be by for from has have he her his i in is it its me my of on or our she
that the their them they this to was were will with you your can could should would about also into than then
there these those what when which who whom how do does did not no yes""".split())


def _tokens(text: str):
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
    toks = list(words)
    toks += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    # crude stemming via prefix features helps match "treatment"/"treatments"/"treating"
    toks += [f"p:{w[:5]}" for w in words if len(w) > 5]
    return toks


class HashingEmbedder:
    dim = config.EMBED_DIM

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in _tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 64) & 1 else -1.0
                out[i, idx] += sign
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class SBertEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype="float32")


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        if os.getenv("EMBEDDING_BACKEND", "hash").lower() == "sbert":
            try:
                _embedder = SBertEmbedder()
            except Exception:
                _embedder = HashingEmbedder()
        else:
            _embedder = HashingEmbedder()
    return _embedder


# --------------------------------------------------------------------------- LangChain adapter
from langchain_core.embeddings import Embeddings  # noqa: E402


class LCEmbeddings(Embeddings):
    """Adapts our embedder to LangChain's Embeddings interface (used by langchain_community FAISS)."""

    def __init__(self):
        self.inner = get_embedder()

    def embed_documents(self, texts):
        return self.inner.embed(list(texts)).tolist()

    def embed_query(self, text):
        return self.inner.embed([text])[0].tolist()
