"""Long-term patient memory (vector store + trace log) and short-term conversation memory."""
from langchain_core.chat_history import InMemoryChatMessageHistory

from . import database as db
from .vectorstore import VectorStore


class PatientMemory:
    """Long-term context per patient. Stores summaries, booked appointments and past interactions."""

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.store = VectorStore(f"patient:{patient_id}")

    def remember(self, text: str, kind: str, run_id: str | None = None) -> bool:
        added = self.store.add(text, {"kind": kind})
        db.add_trace(self.patient_id, f"write:{kind}", text, run_id)
        return added

    def recall(self, query: str, k: int = 3, run_id: str | None = None):
        hits = self.store.search(query, k)
        db.add_trace(self.patient_id, "read", f"query='{query[:120]}' -> {len(hits)} hit(s)", run_id)
        return hits


class ShortTermMemory:
    """Rolling conversation window backed by LangChain's InMemoryChatMessageHistory."""

    def __init__(self, max_turns: int = 6):
        self.history = InMemoryChatMessageHistory()
        self.max_turns = max_turns
        self.last_patient_id: str | None = None

    def add(self, role: str, content: str):
        (self.history.add_user_message if role == "user" else self.history.add_ai_message)(content)
        self.history.messages = self.history.messages[-self.max_turns * 2:]

    def as_text(self) -> str:
        return "\n".join(f"{'user' if m.type == 'human' else 'assistant'}: {str(m.content)[:300]}"
                         for m in self.history.messages[-6:])
