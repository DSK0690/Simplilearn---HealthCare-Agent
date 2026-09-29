"""Agent planning & goal decomposition: LLM planner with a deterministic rule-based fallback."""
import re

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import prompts
from .llm import LLMHandle
from .tools import CONDITION_ALIASES, normalise_condition

VALID_TOOLS = {"identify_patient", "get_medical_history", "book_appointment", "search_medical_info",
               "add_medical_record", "general_answer"}

SPECIALIST_WORDS = {
    "nephrolog": "Nephrology", "cardiolog": "Cardiology", "endocrinolog": "Endocrinology",
    "neurolog": "Neurology", "pulmonolog": "Pulmonology", "general physician": "General Medicine",
    "general medicine": "General Medicine", "physician": "General Medicine",
}
CONDITION_SPECIALTY = {
    "chronic kidney disease": "Nephrology", "hypertension": "Cardiology", "type 2 diabetes": "Endocrinology",
    "asthma": "Pulmonology", "copd": "Pulmonology", "heart failure": "Cardiology",
}

BOOK_RE = re.compile(r"\b(book|appointment|schedule|consult|see a|visit)\b", re.I)
INFO_RE = re.compile(r"\b(treatments?|latest|research|information|symptoms?|guidelines?|causes?|what is|tell me about|"
                     r"search|look up|therap(?:y|ies)|options)\b", re.I)
HISTORY_RE = re.compile(r"\b(history|records?|past diagnos\w*|previous|prior)\b", re.I)
ADD_RE = re.compile(r"\b(add|record|note)\b.*\b(note|record)\b", re.I)


def infer_specialty(query: str) -> str | None:
    q = query.lower()
    for key, spec in SPECIALIST_WORDS.items():
        if key in q:
            return spec
    cond = normalise_condition(q)
    return CONDITION_SPECIALTY.get(cond) if cond else None


def extract_topic(query: str) -> str | None:
    cond = normalise_condition(query)
    if cond:
        return cond
    m = re.search(r"(?:has|have|suffering from|diagnosed with|about)\s+([a-z][a-z\s\-]{3,40}?)(?:[.,?]|\band\b|$)", query, re.I)
    return m.group(1).strip() if m else None


def rule_based_plan(query: str) -> dict:
    steps, goals = [], []

    def add(tool, goal, args=None):
        steps.append({"id": len(steps) + 1, "tool": tool, "goal": goal, "args": args or {}})
        goals.append(goal)

    wants_add = bool(ADD_RE.search(query)) and re.search(r"\b(add|record)\b", query, re.I)
    wants_book = bool(BOOK_RE.search(query))
    wants_info = bool(INFO_RE.search(query))
    wants_hist = bool(HISTORY_RE.search(query))

    if not (wants_add or wants_book or wants_info or wants_hist):
        add("identify_patient", "Identify the patient and context")
        add("general_answer", "Answer the general question")
        return {"goals": goals, "steps": steps, "source": "rules"}

    add("identify_patient", "Identify the patient and context")
    if wants_book or wants_info or wants_hist or wants_add:
        add("get_medical_history", "Retrieve and summarise the patient's medical history")
    if wants_add:
        content = re.split(r"[:]|\bthat\b", query, maxsplit=1)
        text = content[1].strip() if len(content) > 1 else query
        add("add_medical_record", "Store the new record for the patient", {"record_type": "note", "content": text})
    if wants_book:
        add("book_appointment", "Query doctor calendar and book the earliest suitable appointment",
            {"specialty": infer_specialty(query), "reason": query[:120], "preference": query})
    if wants_info:
        add("search_medical_info", "Search trusted sources and summarise the information via the RAG pipeline",
            {"topic": extract_topic(query)})
    return {"goals": goals, "steps": steps, "source": "rules"}


def _validate(plan: dict) -> dict | None:
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list) or not plan["steps"]:
        return None
    steps = []
    for s in plan["steps"]:
        if not isinstance(s, dict) or s.get("tool") not in VALID_TOOLS:
            return None
        args = s.get("args") if isinstance(s.get("args"), dict) else {}
        steps.append({"id": len(steps) + 1, "tool": s["tool"], "goal": str(s.get("goal", s["tool"])), "args": args})
    if steps[0]["tool"] != "identify_patient":
        steps.insert(0, {"id": 0, "tool": "identify_patient", "goal": "Identify the patient and context", "args": {}})
        for i, s in enumerate(steps, 1):
            s["id"] = i
    goals = plan.get("goals") if isinstance(plan.get("goals"), list) else [s["goal"] for s in steps]
    return {"goals": goals, "steps": steps, "source": "llm"}


def make_plan(llm: LLMHandle, query: str, memory_text: str = "", selected_patient: str | None = None) -> dict:
    """Interpret a (possibly multi-step) request and return {'goals','steps','source'}.

    Real LLM: LCEL chain  ChatPromptTemplate | chat model | JsonOutputParser.  Falls back to the rule-based planner
    if the model output is missing / invalid (and always in offline mock mode).
    """
    if not llm.is_mock:
        try:
            prompt = ChatPromptTemplate.from_messages([SystemMessage(content=prompts.PLANNER_SYSTEM),
                                                       ("human", prompts.PLANNER_PROMPT)])
            chain = prompt | llm.chat | JsonOutputParser()
            raw = chain.invoke({"memory": memory_text or "(none)", "selected": selected_patient or "(none)",
                                "query": query})
            plan = _validate(raw)
            if plan:
                return plan
        except Exception:
            pass
    return rule_based_plan(query)
