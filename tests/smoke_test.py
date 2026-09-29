"""End-to-end smoke test (offline mock LLM, temp database). Run: python tests/smoke_test.py"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HEALTHCARE_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ.setdefault("SEARCH_ONLINE", "false")  # keep the test deterministic/offline

from src import database as db, evaluator  # noqa: E402
from src.agent import HealthcareAgent  # noqa: E402
from src.llm import get_llm  # noqa: E402

db.init_db()
agent = HealthcareAgent(get_llm('mock'))

QUERY = ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. "
         "Also, can you summarize latest treatment methods?")
run = agent.run(QUERY, selected_patient_id="P001")
print("PLAN:", [s["tool"] for s in run["plan"]["steps"]], "| source:", run["plan_source"])
for s in run["steps"]:
    print(f"  {s['id']}. {s['tool']:<22} {s['status']}")
print("\n" + run["answer"][:1500])
assert run["status"] == "success", run["status"]
assert [s["tool"] for s in run["steps"]] == ["identify_patient", "get_medical_history", "book_appointment", "search_medical_info"]
assert len(db.list_appointments(patient_id="P001", status="scheduled")) == 1

# idempotency: asking again must not double-book
run2 = agent.run("Book a nephrologist for him", selected_patient_id="P001")
assert len(db.list_appointments(patient_id="P001", status="scheduled")) == 1, "double booking!"

# unknown patient -> graceful skip
db_run = HealthcareAgent(get_llm('mock')).run("Summarize the medical history")
assert db_run["steps"][1]["status"] == "skipped"

res = evaluator.run_suite(agent)
print("\nEVAL SUITE:")
for r in res:
    print(f"  {r['topic']:<24} {r['grade']:<9} rel={r['relevance']} faith={r['faithfulness']} kw={r['keyword_recall']:.0%}")

# ---- ReAct mode (langchain create_agent) exercised with a scripted tool-calling model (no API key needed)
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from src.llm import LLMHandle, _mock_reply  # noqa: E402


class ScriptedToolModel(BaseChatModel):
    script: list = []
    i: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        # The same chat model also powers the tools' internal chains (history summary etc.): answer those offline.
        if not any(m.type == "system" and "autonomous virtual medical assistant" in str(m.content) for m in messages):
            system = "\n".join(str(m.content) for m in messages if m.type == "system")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=_mock_reply(system, str(messages[-1].content))))])
        msg = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


scripted = ScriptedToolModel(script=[
    call("identify_patient", {}, "1"), call("get_medical_history", {}, "2"),
    call("book_appointment", {"specialty": "Cardiology", "preference": "tomorrow morning"}, "3"),
    AIMessage(content="All done: history reviewed and cardiology appointment booked.")])
react_agent = HealthcareAgent(LLMHandle(scripted, "scripted", "scripted"))
r = react_agent.run("Book a cardiologist for Arjun Mehta tomorrow morning", mode="react")
print("\nREACT:", r["plan_source"], [(s["tool"], s["status"]) for s in r["steps"]])
assert r["plan_source"] == "react" and [s["tool"] for s in r["steps"]] == ["identify_patient", "get_medical_history", "book_appointment"]
assert all(s["status"] == "success" for s in r["steps"]) and "All done" in r["answer"]

print("\nMODULE METRICS:", evaluator.module_metrics())
print("\nALL CHECKS PASSED")
