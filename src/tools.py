"""Tools the agent can call: appointment booking, record access, medical information search."""
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import requests

from . import config
from . import database as db

# =========================================================================== appointment tools
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_preference(text: str) -> dict:
    """Extract a preferred date / part-of-day from free text such as 'tomorrow morning' or 'next friday'."""
    t = (text or "").lower()
    pref = {"date": None, "part_of_day": None}
    today = datetime.now().date()
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
    if m:
        pref["date"] = m.group(1)
    elif "tomorrow" in t:
        pref["date"] = (today + timedelta(days=1)).isoformat()
    elif "today" in t:
        pref["date"] = today.isoformat()
    else:
        for i, wd in enumerate(WEEKDAYS):
            if wd in t:
                delta = (i - today.weekday()) % 7 or 7
                pref["date"] = (today + timedelta(days=delta)).isoformat()
                break
    for part in ("morning", "afternoon", "evening"):
        if part in t:
            pref["part_of_day"] = part
    return pref


def book_appointment(patient_id: str, specialty: str, reason: str = "", preference_text: str = "",
                     created_by: str = "agent") -> dict:
    """Find the earliest matching slot with a doctor of `specialty` and book it."""
    if not specialty:
        return {"ok": False, "message": "No medical specialty was provided, so no doctor could be selected."}
    doctors = db.list_doctors(specialty)
    if not doctors:
        avail = ", ".join(db.list_specialties())
        return {"ok": False, "message": f"No {specialty} doctors are registered. Available specialties: {avail}."}

    # Idempotency guard: don't double-book the same specialty for the same patient.
    now = datetime.now().strftime(db.DATE_FMT)
    for a in db.list_appointments(patient_id=patient_id, status="scheduled"):
        if a["specialty"].lower() == specialty.lower() and a["start_time"] > now:
            return {"ok": True, "duplicate": True, "data": a,
                    "message": (f"{a['patient_name']} already has an upcoming {specialty} appointment with "
                                f"{a['doctor_name']} on {a['start_time']} (appointment #{a['id']}); no new booking made.")}

    pref = parse_preference(preference_text)
    note = ""
    slots = db.get_available_slots(specialty=specialty, date=pref["date"], part_of_day=pref["part_of_day"], limit=5)
    if not slots and (pref["date"] or pref["part_of_day"]):
        slots = db.get_available_slots(specialty=specialty, limit=5)
        note = " (your preferred time was unavailable, so the earliest free slot was used)"
    for s in slots:
        appt = db.book_slot(patient_id, s["slot_id"], reason, created_by)
        if appt:
            return {"ok": True, "data": appt,
                    "message": (f"Booked appointment #{appt['id']} for {appt['patient_name']} with {appt['doctor_name']} "
                                f"({appt['specialty']}, {appt['hospital']}) on {appt['start_time']}{note}.")}
    return {"ok": False, "message": f"No free {specialty} slots were found in the next two weeks."}


# =========================================================================== medical information search
LOCAL_KB = {
    "chronic kidney disease": [
        "Chronic kidney disease (CKD) is a long-term condition in which the kidneys gradually lose function, defined by "
        "an eGFR below 60 mL/min/1.73 m2 or markers of kidney damage such as albuminuria for at least three months. "
        "Diabetes and hypertension are the most common causes. Progression is monitored with eGFR and the urine "
        "albumin-to-creatinine ratio.",
        "Treatment of CKD focuses on slowing progression and lowering cardiovascular risk. Blood pressure control is "
        "central, and ACE inhibitors or ARBs are recommended for people with albuminuria. SGLT2 inhibitors are "
        "recommended for many adults with CKD, with or without diabetes, to slow kidney function decline and reduce "
        "heart failure events. In type 2 diabetes with CKD, finerenone and GLP-1 receptor agonists have shown kidney "
        "and cardiovascular benefit. Statins are used to reduce cardiovascular risk.",
        "Supportive care for CKD includes limiting sodium intake, avoiding NSAIDs and other nephrotoxic drugs, "
        "adjusting medication doses to kidney function, and managing anaemia and mineral-bone disorders as the disease "
        "advances. Individualised dietary protein guidance and regular physical activity are commonly advised.",
        "When CKD progresses to kidney failure (eGFR below about 15 mL/min/1.73 m2 or severe symptoms), treatment "
        "options include haemodialysis, peritoneal dialysis, kidney transplantation or conservative kidney management. "
        "For older adults with several conditions, shared decision-making weighs benefits, burdens and personal goals. "
        "Early nephrology referral is advised for rapidly falling eGFR, heavy albuminuria or advanced stages.",
    ],
    "hypertension": [
        "Hypertension (high blood pressure) is a major risk factor for stroke, heart disease and kidney disease and is "
        "diagnosed from repeated readings, ideally confirmed with home or ambulatory monitoring.",
        "Management combines lifestyle measures (reduced salt intake, weight loss, regular exercise, limiting alcohol) "
        "with medication when needed. First-line drug classes include ACE inhibitors, ARBs, calcium channel blockers "
        "and thiazide-type diuretics, often combined to reach the blood pressure target.",
    ],
    "type 2 diabetes": [
        "Type 2 diabetes is characterised by insulin resistance and relative insulin deficiency, leading to raised blood "
        "glucose. HbA1c is used to monitor long-term glucose control.",
        "Treatment includes diet, physical activity and weight management, with metformin commonly used first. "
        "SGLT2 inhibitors and GLP-1 receptor agonists are recommended for people with cardiovascular or kidney disease "
        "because they reduce complications. Regular screening for eye, kidney and foot complications is advised.",
    ],
    "asthma": [
        "Asthma is a chronic inflammatory airway disease causing wheeze, cough, chest tightness and breathlessness. "
        "Triggers include allergens, infections, smoke and exercise.",
        "Treatment relies on inhaled corticosteroids to control inflammation, often combined with a long-acting "
        "bronchodilator, plus a reliever inhaler for symptoms. Written action plans, trigger avoidance and inhaler "
        "technique checks improve control.",
    ],
    "copd": [
        "Chronic obstructive pulmonary disease (COPD) is a progressive lung condition, most often caused by smoking, "
        "marked by breathlessness, chronic cough and airflow limitation on spirometry.",
        "Management includes smoking cessation, bronchodilator inhalers, pulmonary rehabilitation, vaccination and "
        "oxygen therapy for those with severe hypoxaemia. Inhaled corticosteroids are reserved for selected patients "
        "with frequent exacerbations.",
    ],
    "heart failure": [
        "Heart failure occurs when the heart cannot pump enough blood for the body's needs, causing breathlessness, "
        "fatigue and fluid retention.",
        "Guideline-directed therapy for reduced ejection fraction combines ACE inhibitor/ARB/ARNI, beta blockers, "
        "mineralocorticoid receptor antagonists and SGLT2 inhibitors, with diuretics for congestion and lifestyle "
        "measures such as salt and fluid guidance.",
    ],
}
KB_SOURCE = "Local reference KB (summarised clinical guidance; verify with a clinician)"

CONDITION_ALIASES = {
    "chronic kidney disease": ["chronic kidney disease", "ckd", "kidney disease", "renal failure", "kidney failure"],
    "hypertension": ["hypertension", "high blood pressure"],
    "type 2 diabetes": ["type 2 diabetes", "diabetes", "diabetic"],
    "asthma": ["asthma"],
    "copd": ["copd", "chronic obstructive"],
    "heart failure": ["heart failure", "cardiac failure"],
}


def normalise_condition(text: str) -> str | None:
    t = (text or "").lower()
    for canon, aliases in CONDITION_ALIASES.items():
        if any(a in t for a in aliases):
            return canon
    return None


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def medline_search(term: str, retmax: int = 2, timeout: int = 8) -> list[dict]:
    """MedlinePlus Health Topics web service (public, no API key)."""
    r = requests.get("https://wsearch.nlm.nih.gov/ws/query",
                     params={"db": "healthTopics", "term": term, "retmax": retmax}, timeout=timeout)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out = []
    for d in root.iter("document"):
        fields = {c.get("name"): _strip_html("".join(c.itertext())) for c in d.findall("content")}
        text = fields.get("FullSummary") or fields.get("snippet") or ""
        if text:
            out.append({"title": fields.get("title", term), "text": text, "url": d.get("url", ""),
                        "source": "MedlinePlus (U.S. National Library of Medicine)"})
    return out


def local_search(topic: str) -> list[dict]:
    canon = normalise_condition(topic)
    if not canon:
        return []
    return [{"title": canon.title(), "text": t, "url": "", "source": KB_SOURCE} for t in LOCAL_KB.get(canon, [])]


def medical_search(topic: str) -> dict:
    """Fetch documents from trusted sources. Online (MedlinePlus) first, local reference KB as a fallback/complement."""
    docs, errors = [], []
    if config.SEARCH_ONLINE:
        try:
            docs += medline_search(normalise_condition(topic) or topic)
        except Exception as e:  # network blocked / timeout / parse error
            errors.append(f"MedlinePlus unavailable ({type(e).__name__})")
    docs += local_search(topic)
    if not docs:
        return {"ok": False, "docs": [], "message": "No documents found. " + "; ".join(errors), "errors": errors}
    return {"ok": True, "docs": docs, "message": f"Retrieved {len(docs)} document(s).", "errors": errors}


def chunk_text(text: str, size: int = 500) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > size:
            chunks.append(cur.strip())
            cur = ""
        cur += " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
