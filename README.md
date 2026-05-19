# Resilient AI

A production-grade AI agent platform built with **LangGraph**, **FastAPI**, and the **A2A (Agent-to-Agent) protocol**.

## The Problem

Enterprise AI agents break in two common ways:

1. **Tool API churn** — tools are hard-coded at startup; any API change requires redeployment.
2. **LLM provider outages** — a single provider dependency causes total failure during rate limits or downtime.

Additionally, modern multi-agent systems need a **standardised machine-to-machine protocol** so agents can delegate tasks to peer agents without custom integration code.

This gateway solves all three.

---

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │               Resilient AI                  │
                 │                                             │
  REST Client ──►│  POST /v1/stream      ┐                    │
                 │  POST /v1/chat        ├─► ProcessorFactory  │
  A2A Agent  ──►│  POST /agent/a2a/...  ┘       │             │
                 │                               ▼             │
                 │                    LangGraphProcessor       │
                 │                         │         │         │
                 │                  LLMRegistry  StreamProc    │
                 │                  (failover)  (sentence buf) │
                 │                         │                   │
                 │                    LangGraph StateGraph     │
                 │                    ┌────┴────┐              │
                 │                  agent     tools            │
                 │                    └────┬────┘              │
                 │                         └── 4 generic tools │
                 └─────────────────────────────────────────────┘
```

### Key Design Decisions

| Concern | Approach | Tradeoff |
|---|---|---|
| Tool discovery | Bound per-request, not at startup | Flexibility over ~0ms compile latency |
| LLM resilience | 2-provider failover chain (OpenAI → Gemini) | Simplicity over circuit breaker |
| Streaming | Sentence-buffered NDJSON (SSE) | Voice-friendly completeness over minimum latency |
| Multi-surface | Same processor for REST + A2A | Zero logic duplication |
| Session store | In-memory with TTL | Zero deps for demo; swap to Redis in one line |
| Extensibility | Processor registry (Factory + Strategy) | Open/Closed Principle |

---

## Quick Start

### Requirements

- Python 3.11 or higher
- An OpenAI API key (or Gemini API key for failover)

### Run

```bash
chmod +x start.sh stop.sh
./start.sh
```

`start.sh` will:
1. Create `.venv` if it doesn't exist
2. Prepend venv `bin/` (or `Scripts/` on Windows) to `PATH`
3. Install `pip-tools` if not present
4. `pip-compile requirements.in → requirements.txt` (only if `requirements.in` is newer)
5. `pip-sync requirements.txt` — exact venv sync
6. Start FastAPI on `:8000` with `--reload` in `ENV=dev` (default)
7. Health-poll `/health` for up to 30s — prints `Backend running on http://localhost:8000`
8. Start Streamlit on `:8501`

### Dependency management

```bash
# Edit requirements.in (human-maintained, loose pins)
# Then regenerate the locked requirements.txt:
pip-compile requirements.in -o requirements.txt

# Sync your venv exactly to the locked file:
pip-sync requirements.txt

# Upgrade a single package:
pip-compile --upgrade-package langchain requirements.in -o requirements.txt
pip-sync requirements.txt
```

Open **http://localhost:8501** in your browser.

To stop everything:
```bash
./stop.sh
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/session/init` | Create a new chat session |
| `POST` | `/v1/stream` | SSE streaming chat |
| `POST` | `/v1/chat` | Non-streaming chat |
| `POST` | `/agent/a2a/tasks/send` | A2A JSON-RPC 2.0 task submission |
| `GET` | `/agent/a2a/tasks/{id}` | Poll A2A task state + artifacts |
| `GET` | `/agent/a2a/agent-card` | Agent capability descriptor |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Live metrics snapshot |
| `GET` | `/docs` | Interactive API docs (Swagger UI) |

---

## Streamlit Dashboard

Three tabs:

- **💬 Chat** — streaming chat with inline tool call cards and failover badges
- **🔗 A2A Inspector** — fire JSON-RPC 2.0 tasks and watch artifacts arrive
- **📊 Monitoring** — live metrics: requests, tool calls, provider usage, latency

Logs are written to rotating hourly files under `logs/YYYY-MM-DD/` — not streamed in the UI.

---

## Processor Backends

Switch the processing engine via `AGENT_PROCESSOR` in `.env`:

| Key | Status | Description |
|---|---|---|
| `LANGGRAPH` | ✅ Fully functional | Default — LangGraph StateGraph with dynamic tool binding |
| `GOOGLE_ADK` | 🔧 Extension stub | Shows how to wire `google.adk.Agent` |

Adding a new backend: create a subclass of `BaseProcessor`, register it in `processor_factory.py`. Zero other changes needed.

---

## LLM Providers & Failover

The model catalog is a static Python dict in `app/agent/agent_util.py`. The env only selects which provider to prefer — no code changes needed to switch models.

### Model catalog (`agent_util.py`)

| Key | Provider | Model |
|---|---|---|
| `GEMINI_31_FLASH_LITE` | Google GenAI | `gemini-3.1-flash-lite-preview` |
| `GPT4O_MINI` | OpenAI | `gpt-4o-mini` |
| `GEMINI_FLASH` | Google GenAI | `gemini-flash-latest` |
| `GEMINI_25_FLASH` | Google GenAI | `gemini-2.5-flash` |

To add a model: add one line to `llm_provider_map` in `agent_util.py`. To activate it: set `SELECTED_LLM_PROVIDER` to its key.

### Provider selection (`.env`)

```dotenv
# Set preferred provider — moved to the front of the failover chain:
SELECTED_LLM_PROVIDER=GPT4O_MINI
```

### Runtime failover

The full provider chain is built automatically at request time: preferred first, then all other registered providers in map order. On any LLM error (rate limit, auth failure, API error), `LangGraphProcessor` retries with the next provider and emits a `FAILOVER` SSE chunk. The Streamlit chat tab shows a yellow **⚡ Failover** badge when this occurs.

To demo: set `OPENAI_API_KEY=invalid` in `.env`, restart, send a message — the gateway switches to Gemini and the badge appears.

---

## Project Structure

```
app/
├── agent/
│   ├── graph_builder.py        # LangGraph StateGraph + GraphFlowState
│   ├── llm_registry.py         # Multi-provider registry + failover chain
│   ├── runner.py               # AgentRunner: invoke() + stream()
│   ├── stream_processor.py     # NDJSON pipeline + sentence buffer
│   ├── middleware/             # LangChain callback handler
│   └── processors/             # Strategy pattern: LANGGRAPH | ADK | A2A
├── a2a/
│   ├── server.py               # JSON-RPC 2.0 router
│   ├── executor.py             # A2A task → LangGraph stream bridge
│   ├── context_store.py        # context_id → session_id binding
│   └── task_store.py           # In-memory task state
├── tools/                      # 4 generic tools with InjectedState
├── services/                   # Session store + chat history
├── controllers/                # REST API routes
├── middleware/                 # Logger + PII filter
├── config/                     # Settings, logger, metrics
├── models/                     # Pydantic schemas
└── ui/                         # Streamlit dashboard
tests/
└── unit/                       # Unit tests
```

---

## Tools Available

| Tool | Description |
|---|---|
| `calculate` | Safe AST-based expression evaluator |
| `search_knowledge_base` | Mock knowledge base search (ML, Python, microservices, A2A, LangGraph) |
| `get_weather` | Mock weather data for 5 cities |
| `summarize_text` | Extract first two sentences as a summary |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph 1.0, LangChain 0.3 |
| API backend | FastAPI 0.116, Uvicorn 0.47 |
| LLM (primary) | OpenAI gpt-4o-mini |
| LLM (failover) | Google gemini-flash-latest |
| Frontend | Streamlit 1.57 |
| Config | Pydantic v2 BaseSettings |
| Observability | contextvars trace IDs, in-process metrics, structured logging |
| Python | 3.11+ |
