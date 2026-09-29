"""Agentic Healthcare Assistant — Streamlit dashboard.  Run:  streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from src import config, evaluator, tools
from src import database as db
from src.agent import HealthcareAgent
from src.llm import LLMError, get_llm
from src.memory import PatientMemory
from src.vectorstore import HAS_FAISS

st.set_page_config(page_title="Agentic Healthcare Assistant", page_icon="🩺", layout="wide")
db.init_db()

SAMPLE = ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. "
          "Also, can you summarize latest treatment methods?")
SCENARIOS = {
    "Sample scenario (CKD + nephrologist + treatments)": (SAMPLE, "P001"),
    "Medical history & alerts": ("Show me Sunita Devi's medical history and any alerts.", None),
    "Book cardiologist + hypertension info": ("Book a cardiologist for Arjun Mehta tomorrow morning and tell me the latest hypertension treatment.", "P003"),
    "Add a note (attendant action)": ("Add a note: patient reported mild dizziness after the new medication.", "P003"),
    "Disease information only": ("What is asthma? Summarize the treatment options.", "P002"),
    "Failure case: no patient given": ("Please book an appointment with a neurologist.", None),
    "Out-of-scope question": ("Hello, what can you do?", None),
}
STATUS_ICON = {"success": "✅", "failed": "❌", "skipped": "⏭️"}


# ----------------------------------------------------------------------------- shared helpers
def build_llm(provider, key, model):
    try:
        return get_llm(provider, key or None, model or None), None
    except LLMError as e:
        return get_llm("mock"), str(e)


def patient_options(with_none=False):
    ps = db.list_patients()
    opts = {f"{p['name']} ({p['id']})": p["id"] for p in ps}
    return ({"(none)": None} | opts) if with_none else opts


def pick_patient(label="Patient", key=None, with_none=False):
    opts = patient_options(with_none)
    return opts[st.selectbox(label, list(opts), key=key)]


def render_run(run):
    """Plan breakdown + per-step execution trace."""
    with st.expander(f"🧭 Plan & execution trace — run `{run['run_id']}` · {run['status']} · {run['duration_ms']:.0f} ms · planner: {run['plan_source']}"):
        st.markdown("**Sub-goals**")
        for g in run["plan"]["goals"]:
            st.markdown(f"- {g}")
        for s in run["steps"]:
            st.markdown(f"**{s['id']}. `{s['tool']}`** {STATUS_ICON.get(s['status'], '')} {s['status']} · {s['ms']} ms")
            st.caption(f"Goal: {s['goal']} · args: {json.dumps(s['args'], default=str)[:200]}")
        for w in run.get("warnings", []):
            st.warning(w)


def parse_structured(text):
    out = {}
    for line in (text or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🩺 Healthcare Agent")
    page = st.radio("Navigate", [
        "💬 Assistant", "🧑 Patient view", "👨‍⚕️ Doctor view", "📅 Appointments (live)",
        "📚 Medical information", "📝 Medical records", "📊 Evaluation", "🧠 Memory & logs"], label_visibility="collapsed")
    st.divider()
    st.subheader("LLM settings")
    providers = ["mock", "openai", "anthropic"]
    provider = st.selectbox("Provider", providers, index=providers.index(config.LLM_PROVIDER) if config.LLM_PROVIDER in providers else 0,
                            help="'mock' runs fully offline with an extractive stand-in model.")
    api_key = st.text_input("API key (optional if set in .env)", type="password") if provider != "mock" else ""
    model = st.text_input("Model (optional)", value="") if provider != "mock" else ""
    auto_eval = st.toggle("Auto-evaluate answers", value=True)
    mode_label = st.radio("Agent mode", ["Plan-and-execute", "ReAct (autonomous, needs real LLM)"],
                          help="Plan-and-execute: LangChain planner chain + sequential tools (works offline). "
                               "ReAct: langchain create_agent chooses tools itself.")
    st.session_state["agent_mode"] = "react" if mode_label.startswith("ReAct") else "plan_execute"
    llm, err = build_llm(provider, api_key, model)
    if err:
        st.error(f"{err} Falling back to mock mode.")
    st.caption(f"Active: **{llm.name}** {('· ' + llm.model) if llm.model else ''}  \nVector search: **{'FAISS' if HAS_FAISS else 'NumPy (faiss not installed)'}**")
    st.divider()
    if st.button("♻️ Reset demo data"):
        db.reset_demo()
        for k in ("agent", "chat"):
            st.session_state.pop(k, None)
        st.success("Demo data reset.")

if "agent" not in st.session_state:
    st.session_state.agent = HealthcareAgent(llm)
if "chat" not in st.session_state:
    st.session_state.chat = []
agent: HealthcareAgent = st.session_state.agent
agent.llm, agent.auto_eval = llm, auto_eval


# ----------------------------------------------------------------------------- pages
def page_assistant():
    st.header("💬 Virtual medical assistant")
    st.caption("Ask in natural language. The agent plans, calls tools (history, booking, medical search) and reports back.")
    pid = pick_patient("Acting for patient (used if the request doesn't name one)", key="assistant_patient")
    c1, c2, c3 = st.columns(3)
    if c1.button("▶ Run sample scenario", width="stretch"):
        st.session_state.pending = SAMPLE
    if c2.button("Summarise history", width="stretch"):
        st.session_state.pending = "Summarize the medical history"
    if c3.button("Latest hypertension info", width="stretch"):
        st.session_state.pending = "Give me the latest treatment information about hypertension"
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("run"):
                render_run(m["run"])
    query = st.chat_input("e.g. Book a nephrologist for my father and summarise the latest treatments") or st.session_state.pop("pending", None)
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Planning and executing…"):
                run = agent.run(query, selected_patient_id=pid, mode=st.session_state.get("agent_mode", "plan_execute"))
            st.markdown(run["answer"])
            render_run(run)
        st.session_state.chat += [{"role": "user", "content": query}, {"role": "assistant", "content": run["answer"], "run": run}]


def page_patient():
    st.header("🧑 Patient view")
    pid = pick_patient(key="pv_patient")
    p = db.get_patient(pid)
    c = st.columns(4)
    c[0].metric("Name", p["name"]); c[1].metric("Age", p["age"]); c[2].metric("Gender", p["gender"]); c[3].metric("Caregiver", p["caregiver"])
    recs = db.list_records(pid)
    for r in recs:
        if r["record_type"] in ("alert", "allergy"):
            st.warning(f"**{r['record_type'].title()}:** {r['content']}")
    st.subheader("Medical records")
    st.dataframe(pd.DataFrame(recs)[["id", "record_type", "content", "updated_at", "created_by"]] if recs else pd.DataFrame(),
                 width="stretch", hide_index=True)
    if st.button("🤖 Generate AI history summary"):
        with st.spinner("Summarising…"):
            run = agent.run("Summarize the medical history", selected_patient_id=pid)
        st.markdown(run["answer"])
    st.subheader("Appointments")
    appts = db.list_appointments(patient_id=pid)
    st.dataframe(pd.DataFrame(appts)[["id", "start_time", "doctor_name", "specialty", "hospital", "status"]] if appts else pd.DataFrame({"info": ["No appointments"]}),
                 width="stretch", hide_index=True)


def page_doctor():
    st.header("👨‍⚕️ Doctor view")
    docs = {f"{d['name']} — {d['specialty']}": d["id"] for d in db.list_doctors()}
    did = docs[st.selectbox("Doctor", list(docs))]
    appts = db.list_appointments(doctor_id=did)
    upcoming = [a for a in appts if a["status"] == "scheduled"]
    c = st.columns(3)
    c[0].metric("Scheduled", len(upcoming)); c[1].metric("Completed", sum(a["status"] == "completed" for a in appts))
    c[2].metric("Cancelled", sum(a["status"] == "cancelled" for a in appts))
    st.subheader("Upcoming appointments")
    if upcoming:
        st.dataframe(pd.DataFrame(upcoming)[["id", "start_time", "patient_name", "reason", "status"]], width="stretch", hide_index=True)
        col1, col2, col3 = st.columns([2, 1, 1])
        aid = col1.selectbox("Appointment #", [a["id"] for a in upcoming])
        if col2.button("Mark completed"):
            db.update_appointment_status(aid, "completed"); st.rerun()
        if col3.button("Cancel"):
            db.update_appointment_status(aid, "cancelled"); st.rerun()
        pt = st.selectbox("Quick patient history", sorted({a["patient_id"] for a in upcoming}))
        for r in db.list_records(pt):
            st.markdown(f"- **{r['record_type']}** — {r['content']}")
    else:
        st.info("No upcoming appointments for this doctor.")
    st.subheader("Slot utilisation (all doctors, upcoming)")
    util = pd.DataFrame(db.slot_utilisation())
    if not util.empty:
        st.bar_chart(util.set_index("doctor")[["booked", "free"]])


def page_appointments():
    st.header("📅 Appointment tracking")
    live = st.toggle("Live updates (refresh every 5 s)", value=False)

    def board():
        appts = db.list_appointments()
        df = pd.DataFrame(appts)
        m = st.columns(4)
        m[0].metric("Total", len(df))
        for i, s in enumerate(["scheduled", "completed", "cancelled"], 1):
            m[i].metric(s.title(), int((df["status"] == s).sum()) if not df.empty else 0)
        if df.empty:
            st.info("No appointments yet — book one from the Assistant page.")
            return
        flt = st.multiselect("Status filter", ["scheduled", "completed", "cancelled"], default=["scheduled"], key="appt_filter")
        st.dataframe(df[df["status"].isin(flt)][["id", "start_time", "patient_name", "doctor_name", "specialty", "status", "created_by"]],
                     width="stretch", hide_index=True)
        st.caption(f"Last refreshed {db.now_str()}")

    (st.fragment(run_every=5)(board) if live else board)()

    with st.expander("➕ Book / ❌ cancel manually"):
        c1, c2, c3 = st.columns(3)
        pid = c1.selectbox("Patient", list(patient_options()), key="man_pat")
        spec = c2.selectbox("Specialty", db.list_specialties(), key="man_spec")
        pref = c3.text_input("Preference", "tomorrow morning")
        if st.button("Book"):
            r = tools.book_appointment(patient_options()[pid], spec, "Manual booking", pref, created_by="ui")
            (st.success if r["ok"] else st.error)(r["message"])
            db.log_tool("ui", "book_appointment", {"specialty": spec}, r["ok"], 0, r["message"])
        sched = [a for a in db.list_appointments(status="scheduled")]
        if sched:
            aid = st.selectbox("Cancel appointment #", [a["id"] for a in sched])
            if st.button("Cancel appointment"):
                db.update_appointment_status(aid, "cancelled"); st.rerun()


def page_medical_info():
    st.header("📚 Medical information search (RAG)")
    st.caption("Retrieves from MedlinePlus (online) + a local reference KB, chunks & embeds into FAISS, then summarises with citations.")
    c1, c2 = st.columns([3, 1])
    topic = c1.text_input("Disease / condition", "chronic kidney disease")
    pid = c2.selectbox("Patient context", list(patient_options(True)), key="mi_pat")
    pid = patient_options(True)[pid]
    if st.button("🔎 Search & summarise", type="primary") and topic:
        with st.spinner("Retrieving and summarising…"):
            ctx = {"records": db.list_records(pid) if pid else [], "patient": db.get_patient(pid) if pid else None}
            out = agent.search_topic(topic, agent._patient_ctx(ctx) if pid else "", pid)
        if not out["ok"]:
            st.error(out["message"])
        else:
            st.markdown(out["summary"])
            st.markdown("**Sources**")
            for s in out["sources"]:
                st.markdown(f"[{s['idx']}] {s['title']} — {s['source']} {s['url']}")
            if out.get("errors"):
                st.info("; ".join(out["errors"]) + " — local reference KB used.")
            if out.get("eval"):
                e = out["eval"]; st.caption(f"Auto-eval: {e['grade']} · relevance {e['relevance']}/5 · faithfulness {e['faithfulness']}/5")
            with st.expander("Retrieved context (RAG passages)"):
                st.text(out["context"])
    st.subheader("Recent retrieved information")
    rows = db.list_searches(20)
    for r in rows[:5]:
        with st.expander(f"{r['ts']} — {r['topic']} (patient: {r['patient_id'] or '-'})"):
            st.markdown(r["summary"])


def page_records():
    st.header("📝 Medical records (attendant)")
    pid = pick_patient(key="rec_patient")
    recs = db.list_records(pid)
    st.dataframe(pd.DataFrame(recs)[["id", "record_type", "content", "structured", "updated_at", "created_by"]] if recs else pd.DataFrame(),
                 width="stretch", hide_index=True)
    types = ["diagnosis", "medication", "lab", "treatment", "allergy", "alert", "note"]
    t1, t2 = st.tabs(["➕ Add record", "✏️ Update record"])
    with t1:
        with st.form("add_rec", clear_on_submit=True):
            rt = st.selectbox("Type", types, index=6)
            content = st.text_area("Unstructured note / description")
            structured = st.text_area("Structured fields (optional, one `key=value` per line)", placeholder="HbA1c=7.1\nBP=130/80")
            if st.form_submit_button("Save") and content.strip():
                rid = db.add_record(pid, rt, content.strip(), parse_structured(structured), created_by="attendant")
                PatientMemory(pid).remember(f"New {rt} recorded: {content.strip()}", "record")
                st.success(f"Record #{rid} saved and added to patient memory.")
    with t2:
        if recs:
            rid = st.selectbox("Record #", [r["id"] for r in recs])
            cur = next(r for r in recs if r["id"] == rid)
            with st.form("upd_rec"):
                rt = st.selectbox("Type", types, index=types.index(cur["record_type"]) if cur["record_type"] in types else 6)
                content = st.text_area("Content", cur["content"])
                structured = st.text_area("Structured (key=value per line)",
                                          "\n".join(f"{k}={v}" for k, v in json.loads(cur["structured"] or "{}").items()))
                if st.form_submit_button("Update"):
                    db.update_record(rid, rt, content.strip(), parse_structured(structured))
                    PatientMemory(pid).remember(f"Record #{rid} updated ({rt}): {content.strip()}", "record_update")
                    st.success("Record updated."); st.rerun()


def page_evaluation():
    st.header("📊 Model evaluation & monitoring")
    st.caption("QAEvalChain-style LLM-as-judge (heuristic judge in mock mode) + per-module tool success metrics.")
    if st.button("▶ Run evaluation suite (6 diseases)", type="primary"):
        with st.spinner("Running RAG module on the test set and grading…"):
            evaluator.run_suite(agent)
    evals = pd.DataFrame(db.list_evals())
    if not evals.empty:
        suite = evals[evals["kind"] == "suite"].drop_duplicates("topic")
        k = st.columns(4)
        k[0].metric("Answers graded", len(evals))
        k[1].metric("% CORRECT", f"{100 * (evals['grade'] == 'CORRECT').mean():.0f}%")
        k[2].metric("Avg relevance (1-5)", f"{evals['relevance'].mean():.2f}")
        k[3].metric("Avg faithfulness (1-5)", f"{evals['faithfulness'].mean():.2f}")
        if not suite.empty:
            st.subheader("Latest suite results")
            st.dataframe(suite[["topic", "grade", "relevance", "faithfulness", "keyword_recall", "reason"]], width="stretch", hide_index=True)
            st.bar_chart(suite.set_index("topic")[["relevance", "faithfulness"]])
        with st.expander("All evaluation records"):
            st.dataframe(evals, width="stretch", hide_index=True)
    else:
        st.info("No evaluations yet. Run the suite, or use the Assistant / Medical information pages (auto-eval).")
    st.subheader("Agent performance per module")
    mm = pd.DataFrame(evaluator.module_metrics())
    if mm.empty:
        st.info("No tool calls logged yet.")
    else:
        st.dataframe(mm, width="stretch", hide_index=True)
        st.bar_chart(mm.set_index("module")["success_rate"])
        bk = mm[mm["module"] == "book_appointment"]
        if not bk.empty:
            st.metric("Booking success rate", f"{bk.iloc[0]['success_rate']}%")


def page_memory_logs():
    st.header("🧠 Memory, planning & logs")
    t1, t2, t3, t4, t5 = st.tabs(["Memory traces", "Vector memory search", "Agent runs & plans", "Tool logs", "Scenario tester"])
    with t1:
        tr = pd.DataFrame(db.list_traces(200))
        st.dataframe(tr if not tr.empty else pd.DataFrame({"info": ["No memory activity yet"]}), width="stretch", hide_index=True)
    with t2:
        pid = pick_patient(key="mem_patient")
        pm = PatientMemory(pid)
        st.caption(f"{pm.store.count()} memory item(s) stored for this patient.")
        q = st.text_input("Query long-term memory", "kidney treatment")
        if q:
            for h in pm.store.search(q, 5):
                st.markdown(f"**score {h['score']:.2f}** · `{h['meta'].get('kind')}` · {h['created_at']}\n\n{h['text'][:400]}")
    with t3:
        for r in db.list_runs(30):
            with st.expander(f"{r['ts']} · {r['status']} · {r['query'][:80]}"):
                st.caption(f"run {r['run_id']} · patient {r['patient_id']} · {r['duration_ms']:.0f} ms · planner {r['plan_source']}")
                st.json(json.loads(r["plan"]))
                st.markdown(r["response"])
    with t4:
        lg = pd.DataFrame(db.list_tool_logs(500))
        if lg.empty:
            st.info("No tool calls yet.")
        else:
            lg["outcome"] = lg["success"].map({1: "success", 0: "failure"})
            st.bar_chart(lg.pivot_table(index="tool", columns="outcome", values="id", aggfunc="count", fill_value=0))
            st.dataframe(lg[["ts", "run_id", "tool", "outcome", "duration_ms", "message"]], width="stretch", hide_index=True)
    with t5:
        name = st.selectbox("Preset scenario", list(SCENARIOS))
        preset_q, preset_p = SCENARIOS[name]
        q = st.text_area("Request", preset_q, key=f"sc_q_{name}")
        opts = patient_options(True)
        labels = list(opts)
        default = next((i for i, l in enumerate(labels) if opts[l] == preset_p), 0)
        pid = opts[st.selectbox("Selected patient", labels, index=default, key=f"sc_p_{name}")]
        if st.button("Run scenario"):
            with st.spinner("Running…"):
                run = agent.run(q, selected_patient_id=pid, mode=st.session_state.get("agent_mode", "plan_execute"))
            st.markdown(run["answer"])
            render_run(run)
            st.json(run["plan"])


{
    "💬 Assistant": page_assistant, "🧑 Patient view": page_patient, "👨‍⚕️ Doctor view": page_doctor,
    "📅 Appointments (live)": page_appointments, "📚 Medical information": page_medical_info,
    "📝 Medical records": page_records, "📊 Evaluation": page_evaluation, "🧠 Memory & logs": page_memory_logs,
}[page]()
