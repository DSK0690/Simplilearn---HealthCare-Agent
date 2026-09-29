"""LangChain-based agent.

Two execution modes share the same LangChain tools, memory and RAG pipeline:
  * plan_execute (default, works offline): LCEL planner chain -> sequential tool calls -> LCEL answer chain
  * react: an autonomous tool-calling agent built with langchain.agents.create_agent (needs a real chat model)
"""
import json
import re
import time
import uuid

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from . import config, evaluator, planner, prompts, tools
from . import database as db
from .lc_tools import build_tools
from .llm import LLMHandle, get_llm
from .memory import PatientMemory, ShortTermMemory
from .vectorstore import VectorStore

# name -> (system prompt, human template); each becomes:  ChatPromptTemplate | chat model | StrOutputParser
CHAIN_SPECS = {
    "history": (prompts.HISTORY_SYSTEM, prompts.HISTORY_PROMPT),
    "info": (prompts.INFO_SYSTEM, prompts.INFO_PROMPT),
    "final": (prompts.FINAL_SYSTEM, prompts.FINAL_PROMPT),
    "general": (prompts.GENERAL_SYSTEM, "Conversation so far:\n{history}\n\nUser: {query}"),
}


class HealthcareAgent:
    def __init__(self, llm: LLMHandle | None = None, auto_eval: bool = True):
        self.llm = llm or get_llm("mock")
        self.auto_eval = auto_eval
        self.short = ShortTermMemory()
        self._mock = get_llm("mock")

    # ------------------------------------------------------------------ LCEL chains
    @staticmethod
    def _chain(chat, name):
        system, human = CHAIN_SPECS[name]
        prompt = ChatPromptTemplate.from_messages([SystemMessage(content=system), ("human", human)])
        return prompt | chat | StrOutputParser()

    def _run_chain(self, ctx, name, **inputs) -> str:
        """Invoke a chain with the active chat model; on API failure fall back to the offline model."""
        try:
            out = self._chain(self.llm.chat, name).invoke(inputs)
            if out.strip():
                return out
        except Exception as e:
            ctx.setdefault("warnings", []).append(
                f"LLM call failed for '{name}' ({type(e).__name__}); used offline formatter.")
        return self._chain(self._mock.chat, name).invoke(inputs)

    @staticmethod
    def _patient_ctx(ctx) -> str:
        p = ctx.get("patient")
        if not p:
            return "(unknown)"
        dx = [r["content"] for r in ctx.get("records", []) if r["record_type"] == "diagnosis"]
        return f"{p['age']}-year-old {p['gender']}. Known diagnoses: {' '.join(dx) if dx else 'none on file'}"

    # ------------------------------------------------------------------ tool handlers (wrapped by lc_tools)
    def _identify_patient(self, ctx, args):
        q = ctx["query"]
        found, how = None, ""
        m = re.search(r"\bP\d{3}\b", q, re.I)
        if m:
            found, how = db.get_patient(m.group(0).upper()), "patient ID in the request"
        if not found:
            for p in db.list_patients():
                if p["name"].lower() in q.lower():
                    found, how = p, "name in the request"
                    break
        if not found and ctx.get("selected_patient_id"):
            found, how = db.get_patient(ctx["selected_patient_id"]), "patient selected in the UI"
        if not found and self.short.last_patient_id:
            found, how = db.get_patient(self.short.last_patient_id), "previous turn of the conversation"
        if not found:
            return {"ok": False, "message": "I couldn't tell which patient this is about. Please select a patient or mention their name / ID."}
        ctx["patient"] = found
        self.short.last_patient_id = found["id"]
        msg = f"Patient identified: **{found['name']}** ({found['id']}, {found['age']} y, {found['gender']}) via {how}."
        am = re.search(r"(\d{1,3})[- ]year[- ]old", q)
        if am and abs(int(am.group(1)) - found["age"]) > 1:
            msg += f" ⚠️ The request says {am.group(1)} years old but the record says {found['age']}; please verify the patient."
        return {"ok": True, "message": msg, "data": {"patient_id": found["id"]}}

    def _get_medical_history(self, ctx, args):
        p = ctx["patient"]
        records = db.list_records(p["id"])
        ctx["records"] = records
        mem = PatientMemory(p["id"])
        hits = mem.recall(ctx["query"], k=2, run_id=ctx["run_id"])
        mem_text = "\n".join(f"- {h['text'][:300]}" for h in hits) or "(none)"
        rec_text = "\n".join(f"[{r['record_type']}] {r['content']}" for r in records) or "(no records)"
        summary = self._run_chain(ctx, "history", patient=json.dumps({k: p[k] for k in ("name", "age", "gender")}),
                                  records=rec_text, memory=mem_text)
        mem.remember(f"Medical history snapshot ({db.now_str()[:10]}): {summary}", "history_summary", ctx["run_id"])
        alerts = [r["content"] for r in records if r["record_type"] in ("alert", "allergy")]
        return {"ok": True, "message": f"**Medical history summary**\n{summary}",
                "data": {"records": len(records), "alerts": alerts, "memory_hits": [h["text"][:120] for h in hits]}}

    def _book_appointment(self, ctx, args):
        if not ctx.get("records"):  # planner/agent skipped history: load it so specialty can be inferred
            ctx["records"] = db.list_records(ctx["patient"]["id"])
        spec = args.get("specialty") or planner.infer_specialty(ctx["query"])
        if not spec:
            spec = planner.CONDITION_SPECIALTY.get(
                tools.normalise_condition(" ".join(r["content"] for r in ctx["records"])) or "")
        res = tools.book_appointment(ctx["patient"]["id"], spec, reason=args.get("reason") or "",
                                     preference_text=args.get("preference") or ctx["query"])
        if res["ok"]:
            ctx["appointment"] = res["data"]
            a = res["data"]
            PatientMemory(ctx["patient"]["id"]).remember(
                f"Appointment #{a['id']} with {a['doctor_name']} ({a['specialty']}) on {a['start_time']}",
                "appointment", ctx["run_id"])
        return {"ok": res["ok"], "message": ("**Appointment**\n" if res["ok"] else "") + res["message"], "data": res.get("data")}

    def _add_medical_record(self, ctx, args):
        content = (args.get("content") or "").strip()
        if not content:
            return {"ok": False, "message": "No record content was provided."}
        rtype = args.get("record_type") or "note"
        rid = db.add_record(ctx["patient"]["id"], rtype, content, created_by="agent")
        PatientMemory(ctx["patient"]["id"]).remember(f"New {rtype} recorded: {content}", "record", ctx["run_id"])
        return {"ok": True, "message": f"Saved new {rtype} (record #{rid}) to the patient's file.", "data": {"record_id": rid}}

    def _search_medical_info(self, ctx, args):
        topic = args.get("topic")
        if not topic and ctx.get("patient") and not ctx.get("records"):
            ctx["records"] = db.list_records(ctx["patient"]["id"])
        if not topic:
            topic = tools.normalise_condition(" ".join(r["content"] for r in ctx.get("records", [])))
        if not topic:
            return {"ok": False, "message": "I need a disease or condition name to search for."}
        out = self.search_topic(topic, self._patient_ctx(ctx), ctx["patient"]["id"] if ctx.get("patient") else None,
                                ctx["run_id"], ctx)
        if not out["ok"]:
            return {"ok": False, "message": out["message"]}
        srcs = "\n".join(f"[{s['idx']}] {s['title']} — {s['source']}" + (f" ({s['url']})" if s["url"] else "")
                         for s in out["sources"])
        note = f"\n\n_{'; '.join(out['errors'])} — used local reference KB._" if out["errors"] else ""
        return {"ok": True, "data": {"topic": topic, "sources": out["sources"], "eval": out.get("eval")},
                "message": f"**Latest information: {topic}**\n{out['summary']}\n\n**Sources**\n{srcs}{note}"}

    def _general_answer(self, ctx, args):
        text = self._run_chain(ctx, "general", history=self.short.as_text() or "(none)", query=ctx["query"])
        return {"ok": True, "message": text}

    # ------------------------------------------------------------------ RAG pipeline (also used by the eval suite)
    def search_topic(self, topic, patient_ctx="", patient_id=None, run_id=None, ctx=None) -> dict:
        ctx = ctx if ctx is not None else {}
        res = tools.medical_search(topic)                                        # 1) retrieve documents
        if not res["ok"]:
            return {"ok": False, "message": res["message"], "errors": res["errors"]}
        key = tools.normalise_condition(topic) or topic.lower()
        kb = VectorStore("knowledge")
        for d in res["docs"]:                                                     # 2) chunk + embed + index (FAISS)
            for ch in tools.chunk_text(d["text"]):
                kb.add(ch, {"topic": key, "title": d["title"], "source": d["source"], "url": d["url"]})
        hits = [h for h in kb.search(f"{topic} treatment options latest", k=8) if h["meta"].get("topic") == key][:6]
        if not hits:
            return {"ok": False, "message": "No relevant passages could be retrieved.", "errors": res["errors"]}
        src_index, sources, lines = {}, [], []
        for h in hits:
            m = h["meta"]
            k = (m["title"], m["source"], m["url"])
            if k not in src_index:
                src_index[k] = len(src_index) + 1
                sources.append({"idx": src_index[k], "title": m["title"], "source": m["source"], "url": m["url"]})
            lines.append(f"[{src_index[k]}] {h['text']}")
        chunk_txt = "\n".join(lines)
        summary = self._run_chain(ctx, "info", topic=topic, patient=patient_ctx or "(none)", chunks=chunk_txt)  # 3) generate
        db.save_search(patient_id, topic, summary, sources)
        if patient_id:
            PatientMemory(patient_id).remember(f"Retrieved info on {topic} ({db.now_str()[:10]}): {summary[:500]}",
                                               "medical_info", run_id)
        out = {"ok": True, "summary": summary, "sources": sources, "context": chunk_txt, "errors": res["errors"],
               "message": "ok"}
        if self.auto_eval:
            try:
                out["eval"] = evaluator.evaluate_runtime(self.llm, topic, chunk_txt, summary)
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------ execution modes
    @staticmethod
    def _to_step(i, r, goal):
        status = "skipped" if r.get("skipped") else ("success" if r["ok"] else "failed")
        return {"id": i, "tool": r["tool"], "goal": goal, "args": r["args"], "status": status,
                "message": r["message"], "data": r.get("data"), "ms": r["ms"]}

    def _plan_execute(self, ctx, query, selected_patient_id):
        pid = selected_patient_id or self.short.last_patient_id
        mem_text = ""
        if pid:
            mem_text = "\n".join(h["text"][:200] for h in PatientMemory(pid).recall(query, k=2, run_id=ctx["run_id"]))
        plan = planner.make_plan(self.llm, query, mem_text or self.short.as_text(), selected_patient_id)
        toolbox = build_tools(self, ctx)
        results = []
        for step in plan["steps"]:
            n_before = len(ctx.get("tool_results", []))
            try:
                toolbox[step["tool"]].invoke(step["args"])
            except Exception as e:  # invalid arguments from the planner etc.
                msg = f"Error in {step['tool']}: {type(e).__name__}: {str(e)[:160]}"
                db.log_tool(ctx["run_id"], step["tool"], step["args"], False, 0, msg)
                ctx.setdefault("tool_results", []).append({"tool": step["tool"], "args": step["args"], "ok": False,
                                                            "skipped": False, "message": msg, "data": None, "ms": 0})
            if len(ctx.get("tool_results", [])) > n_before:
                results.append(self._to_step(step["id"], ctx["tool_results"][-1], step["goal"]))
        return plan, results, self._compose(ctx, query, results)

    def _react(self, ctx, query, selected_patient_id):
        from langchain.agents import create_agent
        toolbox = build_tools(self, ctx)
        agent = create_agent(self.llm.chat, list(toolbox.values()), system_prompt=prompts.REACT_SYSTEM)
        hint = f"\n\n(Patient selected in the UI: {selected_patient_id})" if selected_patient_id else ""
        out = agent.invoke({"messages": [{"role": "user", "content": query + hint}]}, config={"recursion_limit": 30})
        content = out["messages"][-1].content
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        results = [self._to_step(i, r, "Chosen autonomously by the agent")
                   for i, r in enumerate(ctx.get("tool_results", []), 1)]
        plan = {"goals": ["Autonomous ReAct loop (LangChain create_agent): the model chooses tools step by step"],
                "steps": [{"id": r["id"], "tool": r["tool"], "goal": r["goal"], "args": r["args"]} for r in results],
                "source": "react"}
        return plan, results, f"{content}\n\n---\n_{config.DISCLAIMER}_"

    # ------------------------------------------------------------------ main entry
    def run(self, query: str, selected_patient_id: str | None = None, mode: str = "plan_execute") -> dict:
        run_id = uuid.uuid4().hex[:8]
        t_start = time.time()
        ctx = {"query": query, "selected_patient_id": selected_patient_id, "run_id": run_id, "warnings": []}
        plan = results = answer = None
        if mode == "react":
            if self.llm.is_mock:
                ctx["warnings"].append("ReAct mode needs a real chat model (OpenAI/Anthropic); used plan-and-execute instead.")
            else:
                try:
                    plan, results, answer = self._react(ctx, query, selected_patient_id)
                except Exception as e:
                    ctx["warnings"].append(f"ReAct agent failed ({type(e).__name__}: {str(e)[:120]}); used plan-and-execute.")
                    ctx["tool_results"], ctx["patient"] = [], None
        if plan is None:
            plan, results, answer = self._plan_execute(ctx, query, selected_patient_id)

        ok = [r["status"] for r in results]
        status = "success" if ok and all(s == "success" for s in ok) else ("partial" if "success" in ok else "failed")
        run = {"run_id": run_id, "query": query, "patient_id": (ctx.get("patient") or {}).get("id"), "plan": plan,
               "plan_source": plan["source"], "steps": results, "answer": answer, "status": status,
               "duration_ms": round((time.time() - t_start) * 1000, 1), "warnings": ctx["warnings"]}
        db.log_run(run)
        self.short.add("user", query)
        self.short.add("assistant", answer)
        if run["patient_id"]:
            PatientMemory(run["patient_id"]).remember(
                f"[{db.now_str()[:16]}] Request: {query[:200]} | Outcome: " +
                "; ".join(f"{r['tool']}={r['status']}" for r in results), "interaction", run_id)
        return run

    def _compose(self, ctx, query, results) -> str:
        light = [{"tool": r["tool"], "status": r["status"], "result": r["message"]} for r in results]
        text = self._run_chain(ctx, "final", query=query, results=json.dumps(light, indent=1))
        if any(r["status"] == "skipped" for r in results):
            text += "\n\n_Some steps were skipped because the patient could not be identified._"
        return f"{text}\n\n---\n_{config.DISCLAIMER}_"
