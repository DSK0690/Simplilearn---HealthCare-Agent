"""Persistent vector store: documents + embeddings live in SQLite, search uses FAISS (numpy fallback)."""
import hashlib
import json

import numpy as np

from . import database as db
from .embeddings import LCEmbeddings, get_embedder

try:
    import faiss  # type: ignore  # noqa: F401
    from langchain_community.vectorstores import FAISS as LCFAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    HAS_FAISS = True
except Exception:  # faiss-cpu / langchain-community not installed -> exact numpy search
    LCFAISS = None
    HAS_FAISS = False


class VectorStore:
    """One namespace = one logical index (e.g. 'knowledge' or 'patient:P001')."""

    def __init__(self, namespace: str):
        self.namespace = namespace
        self.embedder = get_embedder()

    # ------------------------------------------------------------------ write
    def add(self, text: str, meta: dict | None = None) -> bool:
        """Add a document. Returns False if an identical text already exists in this namespace."""
        text = text.strip()
        if not text:
            return False
        h = hashlib.sha1(text.encode()).hexdigest()
        emb = self.embedder.embed([text])[0].astype("float32").tobytes()
        with db.conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO vector_docs(namespace,text,text_hash,meta,embedding,created_at) "
                "VALUES (?,?,?,?,?,?)",
                (self.namespace, text, h, json.dumps(meta or {}), emb, db.now_str()))
            return cur.rowcount > 0

    def clear(self):
        with db.conn() as c:
            c.execute("DELETE FROM vector_docs WHERE namespace=?", (self.namespace,))

    # ------------------------------------------------------------------ read
    def _load(self):
        with db.conn() as c:
            rows = c.execute("SELECT text, meta, embedding, created_at FROM vector_docs WHERE namespace=? ORDER BY id",
                             (self.namespace,)).fetchall()
        if not rows:
            return [], None
        mat = np.vstack([np.frombuffer(r["embedding"], dtype="float32") for r in rows])
        docs = [{"text": r["text"], "meta": json.loads(r["meta"] or "{}"), "created_at": r["created_at"]} for r in rows]
        return docs, mat

    def count(self) -> int:
        with db.conn() as c:
            return c.execute("SELECT COUNT(*) FROM vector_docs WHERE namespace=?", (self.namespace,)).fetchone()[0]

    def search(self, query: str, k: int = 4):
        docs, mat = self._load()
        if not docs:
            return []
        q = self.embedder.embed([query]).astype("float32")
        k = min(k, len(docs))
        if HAS_FAISS:
            # LangChain FAISS vector store built from the persisted embeddings (inner product on unit vectors = cosine)
            lc = LCFAISS.from_embeddings(
                [(d["text"], mat[i].tolist()) for i, d in enumerate(docs)], LCEmbeddings(),
                metadatas=[{"i": i} for i in range(len(docs))], distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
            found = lc.similarity_search_with_score_by_vector(q[0].tolist(), k=k)
            pairs = [(d.metadata["i"], score) for d, score in found]
        else:
            sims = mat @ q[0]
            order = np.argsort(-sims)[:k]
            pairs = ((i, sims[i]) for i in order)
        return [{**docs[int(i)], "score": float(s)} for i, s in pairs if int(i) >= 0]
