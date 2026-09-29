"""Shared text heuristics (used by the offline mock model and by the evaluator's fallback judge)."""
import re

from .embeddings import STOPWORDS

_IGNORE = STOPWORDS | {"key", "points", "retrieved", "sources", "source", "patient", "based"}


def terms(text: str) -> set[str]:
    text = re.sub(r"\[\d+\]", " ", text.lower())
    return {w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 3 and w not in _IGNORE}


def stem(ws: set[str]) -> set[str]:
    return {w[:5] for w in ws}


def heuristic_judge(question: str, context: str, answer: str) -> dict:
    q, a, c = stem(terms(question)), stem(terms(answer)), stem(terms(context))
    relevance_ratio = len(q & a) / max(len(q), 1)
    faith_ratio = len(a & c) / max(len(a), 1)
    relevance = round(1 + 4 * min(1.0, relevance_ratio * 1.25), 1)
    faithfulness = round(1 + 4 * faith_ratio, 1)
    grade = "CORRECT" if faith_ratio >= 0.6 and relevance >= 3 else "INCORRECT"
    return {"grade": grade, "relevance": relevance, "faithfulness": faithfulness,
            "reason": f"Heuristic: {relevance_ratio:.0%} of question terms covered, "
                      f"{faith_ratio:.0%} of answer terms grounded in context."}
