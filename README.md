# Agentic Healthcare Assistant for Medical Task Automation

Capstone project — Applied Generative AI Specialisation. A virtual medical assistant built as an **agentic system on the
LangChain framework** (planner chain + tools + memory + RAG + `create_agent`), with an **LLMOps** layer
(`QAEvalChain` evaluation, monitoring, logs) and a **Streamlit** dashboard.

## Quick start (VS Code)

Requires Python 3.10+.

```bash
# 1. open this folder in VS Code, then in the integrated terminal:
python -m venv .venv
.venv\Scripts\activate          # Windows      (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. (optional) real LLM: copy .env.example to .env and set LLM_PROVIDER + API key
#    Without it the app runs fully offline in "mock" mode.

# 3. run
streamlit run app.py
```

Sanity check without the UI: `python tests/smoke_test.py`.

The SQLite database (`data/healthcare.db`) is created and seeded with 3 patients, 7 doctors, appointment slots and
medical records on first launch. Use **Reset demo data** in the sidebar to start over.

## LangChain components used

| Concern | LangChain building block | File |
|---|---|---|
| LLMs | `ChatOpenAI`, `ChatAnthropic` (+ an offline `MockChatModel`, a `BaseChatModel` subclass) | `src/llm.py` |
| Planner | LCEL chain: `ChatPromptTemplate \| chat model \| JsonOutputParser` | `src/planner.py` |
| Tools | `StructuredTool` with Pydantic `args_schema` (6 tools) | `src/lc_tools.py` |
| Autonomous agent | `langchain.agents.create_agent` (ReAct-style tool calling) | `src/agent.py` (`_react`) |
| Prompt chains | summarise history / RAG summary / final answer = LCEL chains with `StrOutputParser` | `src/agent.py` |
| Vector DB | `langchain_community.vectorstores.FAISS` + custom `Embeddings` adapter | `src/vectorstore.py`, `src/embeddings.py` |
| Memory | `InMemoryChatMessageHistory` (short-term) + FAISS per-patient store (long-term) | `src/memory.py` |
| Evaluation | `langchain_classic.evaluation.qa.QAEvalChain` | `src/evaluator.py` |

### Two agent modes (sidebar → *Agent mode*)
1. **Plan-and-execute** (default, works offline): the planner chain breaks the request into sequential sub-goals, each
   executed by a LangChain tool, then an answer chain composes the reply. Falls back to a rule-based planner when the LLM
   returns invalid JSON or in mock mode.
2. **ReAct (autonomous)**: `create_agent` receives the same tools and decides the order itself. Requires a real chat model
   (OpenAI / Anthropic); in mock mode it silently falls back to plan-and-execute.

## Project layout

```
app.py                  Streamlit dashboard (8 views)
src/
  config.py             settings / env vars
  database.py           SQLite: patients, doctors, slots, appointments, records, logs, evals
  llm.py                LangChain chat models (OpenAI / Anthropic / offline mock)
  embeddings.py         hashing embedder (offline) or sentence-transformers; LangChain Embeddings adapter
  vectorstore.py        LangChain FAISS store, documents persisted in SQLite (NumPy fallback)
  memory.py             long-term patient memory + short-term chat history
  tools.py              booking logic, MedlinePlus search, local reference KB, chunking
  lc_tools.py           LangChain StructuredTools wrapping the agent capabilities (+ logging)
  prompts.py            structured prompts per sub-task
  planner.py            goal decomposition (LCEL planner chain + rule-based fallback)
  agent.py              plan-and-execute loop, ReAct mode, RAG pipeline, answer composition
  evaluator.py          QAEvalChain grading, relevance/faithfulness, eval suite, per-module metrics
  heuristics.py         term-overlap heuristics (mock model + judge fallback)
tests/smoke_test.py     end-to-end check: sample scenario, idempotent booking, eval suite, ReAct mode
```

## How it maps to the problem statement

| Requirement | Where |
|---|---|
| 1. Planning & goal decomposition | `planner.py` — sequential sub-goals, each bound to a tool |
| 2. Tools & memory | `lc_tools.py`/`tools.py` (booking, records, MedlinePlus), `database.py` (Doctor Schedule + EHR/Patient DB), FAISS + memory modules |
| 3. Prompt engineering & chaining | `prompts.py`; chain = plan → history summary → booking → RAG summary → final answer; memory lookups injected into prompts |
| 4. Sample execution flow | Assistant page → "Run sample scenario": identify patient → history → book → search & summarise |
| 6. Model evaluation | `evaluator.py` — `QAEvalChain` verdicts, relevance/faithfulness, keyword recall, per-module success rate and latency |
| 7. Data visualisation & UI | Patient view, Doctor view, live Appointments board, Medical information, Evaluation |
| 8. Memory & logs interface | Memory & logs page: memory traces, vector memory search, plans, tool logs, scenario tester |

## Design notes and limitations

- **Mock mode vs real LLM.** Mock mode is extractive and rule-based: deterministic, but not "intelligent". Use OpenAI or
  Anthropic for LLM planning, abstractive summaries, LLM-based judging and ReAct mode.
- **QAEvalChain.** The retrieved passages are used as the reference ("true answer") for grading the generated summary.
  Relevance/faithfulness (1–5) come from an LLM JSON judge (heuristic in mock mode).
- **Medical sources.** Online search uses the public MedlinePlus web service. WHO is not queried live; a local reference KB
  (summarised guidance for 6 conditions) is the fallback. To add WHO or another provider, add a function like
  `medline_search` in `tools.py` and append its documents in `medical_search`.
- **Embeddings.** The default hashing embedder is lexical and offline. For better semantic retrieval set
  `EMBEDDING_BACKEND=sbert` and `pip install sentence-transformers`.
- **Booking safety.** Slots are claimed atomically, and the agent will not double-book the same specialty for a patient.
- **Not a medical device.** Informational/administrative use only; a disclaimer is appended to every answer. Do not use real patient data in this demo.
