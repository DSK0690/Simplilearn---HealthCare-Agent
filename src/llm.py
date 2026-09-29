"""LangChain chat-model layer: ChatOpenAI / ChatAnthropic, or an offline deterministic mock chat model."""
import json
import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from . import config
from .heuristics import heuristic_judge


class LLMError(Exception):
    pass


def extract_json(text: str):
    """Best-effort JSON extraction from an LLM reply (handles code fences and surrounding prose)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- offline mock chat model
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TREATMENT_WORDS = {"treatment", "treat", "therapy", "management", "recommended", "options", "dialysis", "transplant",
                    "inhibitors", "control", "medication", "diet", "care", "guidelines", "reduce", "slow"}


def _mock_history(human: str) -> str:
    m = re.search(r"Patient:\s*(\{.*?\})\s*\n", human, re.S)
    pat = json.loads(m.group(1)) if m else {}
    body = human.split("Records:", 1)[-1].split("Relevant long-term memory:", 1)[0]
    groups: dict[str, list[str]] = {}
    for line in body.strip().splitlines():
        mm = re.match(r"\[(\w+)\]\s*(.*)", line.strip())
        if mm:
            groups.setdefault(mm.group(1), []).append(mm.group(2))
    if not groups:
        return f"No medical records are on file for {pat.get('name', 'this patient')}."
    labels = [("diagnosis", "Diagnoses"), ("medication", "Medications"), ("lab", "Recent labs"),
              ("treatment", "Treatments"), ("note", "Notes")]
    lines = [f"**{pat.get('name')}** ({pat.get('age')} y, {pat.get('gender')})"]
    for key, label in labels:
        if key in groups:
            lines.append(f"- **{label}:** " + " ".join(groups[key]))
    warn = groups.get("allergy", []) + groups.get("alert", [])
    if warn:
        lines.append("- ⚠️ **Alerts & allergies:** " + " ".join(warn))
    return "\n".join(lines)


def _mock_info(human: str) -> str:
    topic = (re.search(r"Topic:\s*(.+)", human) or [None, ""])[1].strip()
    chunk_part = human.split("Source excerpts:", 1)[-1]
    topic_terms = {w for w in re.findall(r"[a-z]+", topic.lower()) if len(w) > 3}
    scored = []
    for order, line in enumerate(chunk_part.strip().splitlines()):
        mm = re.match(r"\[(\d+)\]\s*(.*)", line.strip())
        if not mm:
            continue
        idx, text = mm.group(1), mm.group(2)
        for sent in _SENT_SPLIT.split(text):
            if len(sent) < 40:
                continue
            words = set(re.findall(r"[a-z]+", sent.lower()))
            scored.append((len(words & topic_terms) + 1.5 * len(words & _TREATMENT_WORDS), order, sent.strip(), idx))
    top = sorted(scored, key=lambda x: -x[0])[:6]
    top.sort(key=lambda x: x[1])
    if not top:
        return "No relevant information was retrieved."
    body = "\n".join(f"- {s} [{i}]" for _, _, s, i in top)
    return f"Key points on **{topic}** from the retrieved sources:\n{body}"


def _mock_final(human: str) -> str:
    raw = human.split("Step results (JSON):", 1)[-1]
    data = extract_json("{\"r\": " + raw.strip() + "}") or {}
    msgs = [r["result"] for r in data.get("r", []) if r.get("status") != "skipped"]
    return "\n\n".join(msgs)


def _mock_grade(human: str) -> str:
    """Grades like QAEvalChain: CORRECT / INCORRECT from term overlap with the reference."""
    block = human.rsplit("QUESTION:", 1)[-1]
    q = block.split("STUDENT ANSWER:", 1)[0].split("CONTEXT:", 1)[0].strip()
    rest = block.split("STUDENT ANSWER:", 1)[-1] if "STUDENT ANSWER:" in block else block
    student = re.split(r"TRUE ANSWER:|CONTEXT:|GRADE:", rest)[0].strip()
    ref = ""
    for marker in ("TRUE ANSWER:", "CONTEXT:"):
        if marker in block:
            ref = block.split(marker, 1)[-1].split("GRADE:", 1)[0]
            break
    return heuristic_judge(q, ref, student)["grade"]


def _mock_reply(system: str, human: str) -> str:
    if "either CORRECT or INCORRECT" in system + human:
        return _mock_grade(system + "\n" + human)
    if "clinical administrative assistant" in system:
        return _mock_history(human)
    if "medical information assistant" in system:
        return _mock_info(human)
    if "friendly virtual medical assistant" in system:
        return _mock_final(human)
    if "virtual medical assistant that can book" in system:
        return ("I can help with four things: booking appointments, managing and summarising medical records, "
                "and looking up up-to-date disease information. Try: *\"Book a nephrologist for my father and "
                "summarise the latest treatment methods\"*.")
    return ""  # planner / JSON judge -> callers fall back to rule-based / heuristic logic


class MockChatModel(BaseChatModel):
    """Deterministic, extractive stand-in so the whole app works without any API key.

    It does NOT reason: it formats / extracts from the prompt text. It plugs into the same LangChain chains as a
    real chat model.
    """

    @property
    def _llm_type(self) -> str:
        return "mock-extractive"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = "\n".join(str(m.content) for m in messages if m.type == "system")
        human = str(messages[-1].content) if messages else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=_mock_reply(system, human)))])

    def bind_tools(self, tools, **kwargs):  # the mock never calls tools
        return self


# --------------------------------------------------------------------------- handle + factory
@dataclass
class LLMHandle:
    chat: BaseChatModel
    name: str
    model: str
    is_mock: bool = False


def get_llm(provider: str | None = None, api_key: str | None = None, model: str | None = None) -> LLMHandle:
    provider = (provider or config.LLM_PROVIDER).lower()
    model = model or config.LLM_MODEL
    if provider == "openai":
        key = api_key or config.OPENAI_API_KEY
        if not key:
            raise LLMError("OPENAI_API_KEY is missing.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise LLMError("Install langchain-openai: pip install langchain-openai") from e
        m = model or config.MODEL_DEFAULTS["openai"]
        return LLMHandle(ChatOpenAI(model=m, api_key=key, temperature=0.2, max_retries=1, timeout=60), "openai", m)
    if provider == "anthropic":
        key = api_key or config.ANTHROPIC_API_KEY
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is missing.")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise LLMError("Install langchain-anthropic: pip install langchain-anthropic") from e
        m = model or config.MODEL_DEFAULTS["anthropic"]
        return LLMHandle(ChatAnthropic(model=m, api_key=key, temperature=0.2, max_tokens=1500, max_retries=1, timeout=60),
                         "anthropic", m)
    return LLMHandle(MockChatModel(), "mock", "offline-extractive", True)
