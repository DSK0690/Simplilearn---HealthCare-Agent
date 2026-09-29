"""Model evaluation: LangChain QAEvalChain grading + relevance/faithfulness scores + per-module performance metrics."""
import re
from collections import defaultdict

from langchain_classic.evaluation.qa import QAEvalChain
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import database as db
from . import prompts
from .heuristics import heuristic_judge
from .llm import LLMHandle

# Ground-truth style test set for the medical-information (RAG) module.
EVAL_CASES = [
    {"topic": "chronic kidney disease", "question": "What are the main treatment options for chronic kidney disease?",
     "keywords": ["blood pressure", "sglt2", "dialysis", "transplant"]},
    {"topic": "hypertension", "question": "How is hypertension managed?",
     "keywords": ["lifestyle", "ace inhibitors", "blood pressure"]},
    {"topic": "type 2 diabetes", "question": "What treatments are used for type 2 diabetes?",
     "keywords": ["metformin", "sglt2", "hba1c"]},
    {"topic": "asthma", "question": "How is asthma treated?",
     "keywords": ["inhaled corticosteroids", "reliever", "trigger"]},
    {"topic": "copd", "question": "What is the management of COPD?",
     "keywords": ["smoking", "bronchodilator", "rehabilitation"]},
    {"topic": "heart failure", "question": "What is guideline-directed therapy for heart failure?",
     "keywords": ["beta blockers", "sglt2", "diuretics"]},
]


def _qa_eval_grade(llm: LLMHandle, question: str, context: str, answer: str) -> str | None:
    """Grade with LangChain's QAEvalChain (the retrieved context is the reference/true answer)."""
    try:
        chain = QAEvalChain.from_llm(llm.chat)
        out = chain.evaluate([{"query": question, "answer": context[:4000]}], [{"result": answer}])[0]
        text = str(next(iter(out.values()))).upper()
        if "INCORRECT" in text:
            return "INCORRECT"
        if "CORRECT" in text:
            return "CORRECT"
    except Exception:
        pass
    return None


def judge(llm: LLMHandle, question: str, context: str, answer: str) -> dict:
    """Grade an answer: QAEvalChain verdict + 1-5 relevance/faithfulness (LLM JSON judge, heuristic fallback)."""
    res = heuristic_judge(question, context, answer)
    grade = _qa_eval_grade(llm, question, context, answer)
    if grade:
        res["grade"] = grade
    if not llm.is_mock:
        try:
            prompt = ChatPromptTemplate.from_messages([SystemMessage(content=prompts.JUDGE_SYSTEM),
                                                       ("human", prompts.JUDGE_PROMPT)])
            j = (prompt | llm.chat | JsonOutputParser()).invoke(
                {"question": question, "context": context[:4000], "answer": answer})
            res.update(relevance=float(j.get("relevance", res["relevance"])),
                       faithfulness=float(j.get("faithfulness", res["faithfulness"])),
                       reason=str(j.get("reason", res["reason"])))
        except Exception:
            pass
    return res


def keyword_recall(answer: str, keywords: list[str]) -> float:
    a = answer.lower()
    return sum(1 for k in keywords if k in a) / max(len(keywords), 1)


def evaluate_runtime(llm, topic: str, context: str, answer: str) -> dict:
    """Automatic per-request evaluation of a generated summary (logged as kind='runtime')."""
    q = f"What are the treatment options for {topic}?"
    r = judge(llm, q, context, answer)
    db.save_eval("runtime", topic, q, r["grade"], r["relevance"], r["faithfulness"], None, r["reason"])
    return r


def run_suite(agent, patient_id=None) -> list[dict]:
    """Run the RAG module on every EVAL_CASE, grade with judge + keyword recall, persist results."""
    results = []
    for case in EVAL_CASES:
        out = agent.search_topic(case["topic"], patient_ctx="", patient_id=patient_id)
        if not out["ok"]:
            db.save_eval("suite", case["topic"], case["question"], "INCORRECT", 0, 0, 0, out["message"])
            results.append({**case, "grade": "INCORRECT", "relevance": 0, "faithfulness": 0, "keyword_recall": 0,
                            "reason": out["message"]})
            continue
        j = judge(agent.llm, case["question"], out["context"], out["summary"])
        kr = keyword_recall(out["summary"], case["keywords"])
        db.save_eval("suite", case["topic"], case["question"], j["grade"], j["relevance"], j["faithfulness"], kr,
                     j["reason"])
        results.append({**case, **j, "keyword_recall": kr})
    return results


def module_metrics() -> list[dict]:
    """Aggregate tool logs into per-module success rate and latency."""
    agg = defaultdict(lambda: {"calls": 0, "ok": 0, "ms": 0.0})
    for r in db.list_tool_logs(5000):
        a = agg[r["tool"]]
        a["calls"] += 1
        a["ok"] += r["success"]
        a["ms"] += r["duration_ms"] or 0
    return [{"module": k, "calls": v["calls"], "success_rate": round(100 * v["ok"] / v["calls"], 1),
             "avg_latency_ms": round(v["ms"] / v["calls"], 1)} for k, v in sorted(agg.items())]
