"""The agent's capabilities exposed as LangChain StructuredTools (usable by the planner loop and by create_agent)."""
import time
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from . import database as db

NEEDS_PATIENT = {"get_medical_history", "book_appointment", "add_medical_record"}


class NoArgs(BaseModel):
    """No arguments: the tool uses the current request context."""


class BookArgs(BaseModel):
    specialty: Optional[str] = Field(None, description="Nephrology, Cardiology, Endocrinology, General Medicine, Neurology or Pulmonology")
    reason: str = Field("", description="Short reason for the visit")
    preference: str = Field("", description="Preferred time in words, e.g. 'tomorrow morning'. Empty = earliest slot")


class SearchArgs(BaseModel):
    topic: Optional[str] = Field(None, description="Disease or condition name to look up, e.g. 'chronic kidney disease'")


class AddRecordArgs(BaseModel):
    content: str = Field(..., description="Text of the note/record to store")
    record_type: str = Field("note", description="diagnosis, medication, lab, treatment, allergy, alert or note")


def build_tools(agent, ctx: dict) -> dict[str, StructuredTool]:
    """Create per-request tools bound to `ctx`. Every call is timed, logged to SQLite and appended to ctx['tool_results']."""

    def wrap(name, handler):
        def _run(**kwargs) -> str:
            t0 = time.time()
            if name in NEEDS_PATIENT and not ctx.get("patient"):
                r = {"ok": False, "skipped": True,
                     "message": "Skipped: patient not identified. Call identify_patient first."}
            else:
                try:
                    r = handler(ctx, kwargs)
                except Exception as e:  # a failing tool must never crash the agent
                    r = {"ok": False, "message": f"Error in {name}: {type(e).__name__}: {e}"}
            ms = (time.time() - t0) * 1000
            if not r.get("skipped"):
                db.log_tool(ctx["run_id"], name, kwargs, r["ok"], ms, r["message"])
            ctx.setdefault("tool_results", []).append({
                "tool": name, "args": kwargs, "ok": r["ok"], "skipped": r.get("skipped", False),
                "message": r["message"], "data": r.get("data"), "ms": round(ms, 1)})
            return r["message"]
        return _run

    specs = [
        ("identify_patient", "Work out which patient the request is about (from ID, name, UI selection or conversation). Always call first.", NoArgs, agent._identify_patient),
        ("get_medical_history", "Load the patient's records and long-term memory and return a summary incl. alerts/allergies.", NoArgs, agent._get_medical_history),
        ("book_appointment", "Find the earliest free slot with a doctor of the given specialty and book it for the patient.", BookArgs, agent._book_appointment),
        ("search_medical_info", "Retrieve trusted disease information (MedlinePlus + reference KB) via RAG and summarise it with citations.", SearchArgs, agent._search_medical_info),
        ("add_medical_record", "Store a new note/record in the patient's file.", AddRecordArgs, agent._add_medical_record),
        ("general_answer", "Answer a general question outside the other tools' scope.", NoArgs, agent._general_answer),
    ]
    return {n: StructuredTool.from_function(func=wrap(n, h), name=n, description=d, args_schema=s)
            for n, d, s, h in specs}
