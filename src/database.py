"""SQLite persistence layer: patients, doctors, slots, appointments, records, memory, logs, evaluations."""
import json
import random
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients(
    id TEXT PRIMARY KEY, name TEXT NOT NULL, age INTEGER, gender TEXT,
    phone TEXT, caregiver TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS doctors(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, specialty TEXT NOT NULL, hospital TEXT);
CREATE TABLE IF NOT EXISTS slots(
    id INTEGER PRIMARY KEY AUTOINCREMENT, doctor_id INTEGER NOT NULL, start_time TEXT NOT NULL,
    booked INTEGER DEFAULT 0, UNIQUE(doctor_id, start_time));
CREATE TABLE IF NOT EXISTS appointments(
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, doctor_id INTEGER, slot_id INTEGER,
    start_time TEXT, reason TEXT, status TEXT DEFAULT 'scheduled', created_at TEXT, created_by TEXT);
CREATE TABLE IF NOT EXISTS records(
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, record_type TEXT, content TEXT,
    structured TEXT, created_at TEXT, updated_at TEXT, created_by TEXT);
CREATE TABLE IF NOT EXISTS vector_docs(
    id INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT, text TEXT, text_hash TEXT, meta TEXT,
    embedding BLOB, created_at TEXT, UNIQUE(namespace, text_hash));
CREATE TABLE IF NOT EXISTS memory_traces(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, patient_id TEXT, kind TEXT, content TEXT, run_id TEXT);
CREATE TABLE IF NOT EXISTS tool_logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, run_id TEXT, tool TEXT, args TEXT,
    success INTEGER, duration_ms REAL, message TEXT);
CREATE TABLE IF NOT EXISTS agent_runs(
    run_id TEXT PRIMARY KEY, ts TEXT, patient_id TEXT, query TEXT, plan TEXT, plan_source TEXT,
    response TEXT, status TEXT, duration_ms REAL);
CREATE TABLE IF NOT EXISTS med_searches(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, patient_id TEXT, topic TEXT, summary TEXT, sources TEXT);
CREATE TABLE IF NOT EXISTS eval_results(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, topic TEXT, question TEXT, grade TEXT,
    relevance REAL, faithfulness REAL, keyword_recall REAL, reason TEXT);
"""

DATE_FMT = "%Y-%m-%d %H:%M"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def conn():
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def _rows(cur):
    return [dict(r) for r in cur.fetchall()]


# --------------------------------------------------------------------------- init / seed
def init_db(seed: bool = True):
    with conn() as c:
        c.executescript(SCHEMA)
        n = c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    if seed and n == 0:
        seed_data()
    ensure_future_slots()


def reset_demo():
    with conn() as c:
        for t in ["patients", "doctors", "slots", "appointments", "records", "vector_docs",
                  "memory_traces", "tool_logs", "agent_runs", "med_searches", "eval_results"]:
            c.execute(f"DELETE FROM {t}")
        c.execute("DELETE FROM sqlite_sequence")
    seed_data()
    ensure_future_slots()


def seed_data():
    ts = now_str()
    patients = [
        ("P001", "Ramesh Kumar", 70, "Male", "+91-90000-11111", "Anita Kumar (daughter)"),
        ("P002", "Sunita Devi", 58, "Female", "+91-90000-22222", "Vikram Devi (son)"),
        ("P003", "Arjun Mehta", 45, "Male", "+91-90000-33333", "Self"),
    ]
    doctors = [
        ("Dr. Meera Iyer", "Nephrology", "City General Hospital"),
        ("Dr. Arun Shah", "Nephrology", "Lakeside Kidney Centre"),
        ("Dr. Priya Nair", "Cardiology", "City General Hospital"),
        ("Dr. Karthik Rao", "Endocrinology", "Lakeside Kidney Centre"),
        ("Dr. Sana Khan", "General Medicine", "City General Hospital"),
        ("Dr. Rohan Verma", "Neurology", "Metro Neuro Institute"),
        ("Dr. Leela Menon", "Pulmonology", "Metro Neuro Institute"),
    ]
    records = [
        ("P001", "diagnosis", "Chronic kidney disease (CKD) stage 3b, diagnosed 2022. eGFR 38 mL/min/1.73m2.", {"eGFR": 38, "stage": "3b"}),
        ("P001", "diagnosis", "Hypertension since 2010. Type 2 diabetes mellitus since 2015.", {}),
        ("P001", "medication", "Telmisartan 40 mg daily; Metformin dose reduced to 500 mg twice daily; Atorvastatin 20 mg nightly.", {}),
        ("P001", "lab", "Creatinine 1.9 mg/dL, HbA1c 7.4%, Potassium 4.8 mmol/L (last month).", {"creatinine": 1.9, "HbA1c": 7.4}),
        ("P001", "treatment", "Low-sodium diet advised; fluid intake monitoring; quarterly nephrology follow-up.", {}),
        ("P001", "allergy", "Allergic to Penicillin (rash).", {}),
        ("P001", "alert", "Avoid NSAIDs (ibuprofen, diclofenac) because of reduced kidney function.", {}),
        ("P002", "diagnosis", "Bronchial asthma, well controlled on inhaled therapy.", {}),
        ("P002", "medication", "Budesonide-formoterol inhaler twice daily; salbutamol as needed.", {}),
        ("P002", "note", "Annual flu vaccination advised. No hospital admissions in last 2 years.", {}),
        ("P003", "diagnosis", "Essential hypertension diagnosed 2023.", {}),
        ("P003", "medication", "Amlodipine 5 mg once daily.", {}),
        ("P003", "alert", "Family history of premature coronary artery disease.", {}),
    ]
    with conn() as c:
        for p in patients:
            c.execute("INSERT INTO patients VALUES (?,?,?,?,?,?,?)", (*p, ts))
        for d in doctors:
            c.execute("INSERT INTO doctors(name,specialty,hospital) VALUES (?,?,?)", d)
        for pid, rtype, content, st in records:
            c.execute("INSERT INTO records(patient_id,record_type,content,structured,created_at,updated_at,created_by) "
                      "VALUES (?,?,?,?,?,?,?)", (pid, rtype, content, json.dumps(st), ts, ts, "seed"))


def ensure_future_slots(days: int = 14):
    """Make sure every doctor has bookable slots for the next `days` days (Sundays off)."""
    now = datetime.now()
    hours = [9, 10, 11, 12, 14, 15, 16, 17]
    with conn() as c:
        docs = [r["id"] for r in c.execute("SELECT id FROM doctors")]
        for doc in docs:
            for d in range(days):
                day = (now + timedelta(days=d)).replace(minute=0, second=0, microsecond=0)
                if day.weekday() == 6:
                    continue
                for h in hours:
                    st = day.replace(hour=h)
                    if st <= now:
                        continue
                    key = st.strftime(DATE_FMT)
                    exists = c.execute("SELECT 1 FROM slots WHERE doctor_id=? AND start_time=?", (doc, key)).fetchone()
                    if exists:
                        continue
                    # Deterministic pseudo-random: some slots are already taken by other patients.
                    rnd = random.Random(f"{doc}-{key}").random()
                    c.execute("INSERT OR IGNORE INTO slots(doctor_id,start_time,booked) VALUES (?,?,?)",
                              (doc, key, 1 if rnd < 0.3 else 0))


# --------------------------------------------------------------------------- patients / doctors
def list_patients():
    with conn() as c:
        return _rows(c.execute("SELECT * FROM patients ORDER BY id"))


def get_patient(pid):
    with conn() as c:
        r = c.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
        return dict(r) if r else None


def list_doctors(specialty=None):
    with conn() as c:
        if specialty:
            return _rows(c.execute("SELECT * FROM doctors WHERE LOWER(specialty)=LOWER(?) ORDER BY id", (specialty,)))
        return _rows(c.execute("SELECT * FROM doctors ORDER BY specialty, name"))


def get_doctor(did):
    with conn() as c:
        r = c.execute("SELECT * FROM doctors WHERE id=?", (did,)).fetchone()
        return dict(r) if r else None


def list_specialties():
    with conn() as c:
        return [r[0] for r in c.execute("SELECT DISTINCT specialty FROM doctors ORDER BY specialty")]


# --------------------------------------------------------------------------- slots / appointments
def get_available_slots(specialty=None, doctor_id=None, date=None, part_of_day=None, limit=5):
    after = (datetime.now() + timedelta(hours=1)).strftime(DATE_FMT)
    q = ("SELECT s.id AS slot_id, s.start_time, d.id AS doctor_id, d.name AS doctor_name, d.specialty, d.hospital "
         "FROM slots s JOIN doctors d ON d.id=s.doctor_id WHERE s.booked=0 AND s.start_time>?")
    args = [after]
    if specialty:
        q += " AND LOWER(d.specialty)=LOWER(?)"
        args.append(specialty)
    if doctor_id:
        q += " AND d.id=?"
        args.append(doctor_id)
    if date:
        q += " AND s.start_time LIKE ?"
        args.append(f"{date}%")
    if part_of_day == "morning":
        q += " AND substr(s.start_time,12,2) < '12'"
    elif part_of_day == "afternoon":
        q += " AND substr(s.start_time,12,2) >= '12' AND substr(s.start_time,12,2) < '17'"
    elif part_of_day == "evening":
        q += " AND substr(s.start_time,12,2) >= '17'"
    q += " ORDER BY s.start_time LIMIT ?"
    args.append(limit)
    with conn() as c:
        return _rows(c.execute(q, args))


def book_slot(patient_id, slot_id, reason="", created_by="agent"):
    """Atomically claim a slot. Returns appointment dict or None if the slot was taken."""
    with conn() as c:
        cur = c.execute("UPDATE slots SET booked=1 WHERE id=? AND booked=0", (slot_id,))
        if cur.rowcount == 0:
            return None
        s = c.execute("SELECT * FROM slots WHERE id=?", (slot_id,)).fetchone()
        cur = c.execute(
            "INSERT INTO appointments(patient_id,doctor_id,slot_id,start_time,reason,status,created_at,created_by) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (patient_id, s["doctor_id"], slot_id, s["start_time"], reason, "scheduled", now_str(), created_by))
        aid = cur.lastrowid
    return get_appointment(aid)


def get_appointment(aid):
    with conn() as c:
        r = c.execute(_APPT_SQL + " WHERE a.id=?", (aid,)).fetchone()
        return dict(r) if r else None


_APPT_SQL = ("SELECT a.id, a.patient_id, p.name AS patient_name, a.doctor_id, d.name AS doctor_name, d.specialty, "
             "d.hospital, a.start_time, a.reason, a.status, a.created_at, a.created_by "
             "FROM appointments a JOIN patients p ON p.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id")


def list_appointments(patient_id=None, doctor_id=None, status=None):
    q, args, where = _APPT_SQL, [], []
    if patient_id:
        where.append("a.patient_id=?"); args.append(patient_id)
    if doctor_id:
        where.append("a.doctor_id=?"); args.append(doctor_id)
    if status:
        where.append("a.status=?"); args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY a.start_time"
    with conn() as c:
        return _rows(c.execute(q, args))


def update_appointment_status(aid, status):
    with conn() as c:
        a = c.execute("SELECT slot_id FROM appointments WHERE id=?", (aid,)).fetchone()
        c.execute("UPDATE appointments SET status=? WHERE id=?", (status, aid))
        if status == "cancelled" and a:
            c.execute("UPDATE slots SET booked=0 WHERE id=?", (a["slot_id"],))


def slot_utilisation():
    with conn() as c:
        return _rows(c.execute(
            "SELECT d.name AS doctor, SUM(s.booked) AS booked, COUNT(*)-SUM(s.booked) AS free "
            "FROM slots s JOIN doctors d ON d.id=s.doctor_id WHERE s.start_time>=? GROUP BY d.id ORDER BY d.name",
            (datetime.now().strftime(DATE_FMT),)))


# --------------------------------------------------------------------------- medical records
def list_records(patient_id):
    with conn() as c:
        return _rows(c.execute("SELECT * FROM records WHERE patient_id=? ORDER BY id", (patient_id,)))


def add_record(patient_id, record_type, content, structured=None, created_by="attendant"):
    ts = now_str()
    with conn() as c:
        cur = c.execute(
            "INSERT INTO records(patient_id,record_type,content,structured,created_at,updated_at,created_by) "
            "VALUES (?,?,?,?,?,?,?)",
            (patient_id, record_type, content, json.dumps(structured or {}), ts, ts, created_by))
        return cur.lastrowid


def update_record(record_id, record_type, content, structured=None):
    with conn() as c:
        c.execute("UPDATE records SET record_type=?, content=?, structured=?, updated_at=? WHERE id=?",
                  (record_type, content, json.dumps(structured or {}), now_str(), record_id))


# --------------------------------------------------------------------------- logs / traces
def add_trace(patient_id, kind, content, run_id=None):
    with conn() as c:
        c.execute("INSERT INTO memory_traces(ts,patient_id,kind,content,run_id) VALUES (?,?,?,?,?)",
                  (now_str(), patient_id, kind, content[:2000], run_id))


def list_traces(limit=200):
    with conn() as c:
        return _rows(c.execute("SELECT * FROM memory_traces ORDER BY id DESC LIMIT ?", (limit,)))


def log_tool(run_id, tool, args, success, duration_ms, message):
    with conn() as c:
        c.execute("INSERT INTO tool_logs(ts,run_id,tool,args,success,duration_ms,message) VALUES (?,?,?,?,?,?,?)",
                  (now_str(), run_id, tool, json.dumps(args, default=str)[:1000], int(bool(success)),
                   round(duration_ms, 1), (message or "")[:500]))


def list_tool_logs(limit=500):
    with conn() as c:
        return _rows(c.execute("SELECT * FROM tool_logs ORDER BY id DESC LIMIT ?", (limit,)))


def log_run(run):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO agent_runs VALUES (?,?,?,?,?,?,?,?,?)",
                  (run["run_id"], now_str(), run.get("patient_id"), run["query"],
                   json.dumps(run["plan"], default=str), run.get("plan_source", ""),
                   run.get("answer", ""), run.get("status", ""), run.get("duration_ms", 0)))


def list_runs(limit=100):
    with conn() as c:
        return _rows(c.execute("SELECT * FROM agent_runs ORDER BY ts DESC LIMIT ?", (limit,)))


def save_search(patient_id, topic, summary, sources):
    with conn() as c:
        c.execute("INSERT INTO med_searches(ts,patient_id,topic,summary,sources) VALUES (?,?,?,?,?)",
                  (now_str(), patient_id, topic, summary, json.dumps(sources)))


def list_searches(limit=50):
    with conn() as c:
        return _rows(c.execute("SELECT * FROM med_searches ORDER BY id DESC LIMIT ?", (limit,)))


def save_eval(kind, topic, question, grade, relevance, faithfulness, keyword_recall, reason):
    with conn() as c:
        c.execute("INSERT INTO eval_results(ts,kind,topic,question,grade,relevance,faithfulness,keyword_recall,reason) "
                  "VALUES (?,?,?,?,?,?,?,?,?)",
                  (now_str(), kind, topic, question, grade, relevance, faithfulness, keyword_recall, reason[:500]))


def list_evals(limit=300):
    with conn() as c:
        return _rows(c.execute("SELECT * FROM eval_results ORDER BY id DESC LIMIT ?", (limit,)))
