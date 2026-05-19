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
  A2A Agent  ──►│  POST /agent/a2a      ┘       │             │
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
                 │              dynamic tool binding           │
                 │           (MCP-extensible at runtime)       │
                 └─────────────────────────────────────────────┘
```

### Key Design Decisions

| Concern | Approach | Tradeoff |
|---|---|---|
| Tool discovery | Bound per-request inside `agent_node`, not at startup | Enables runtime MCP tool injection without graph recompilation |
| LLM resilience | Full failover chain — preferred provider first, then all others in map order | Automatic recovery over manual circuit breaker |
| Streaming | Sentence-buffered NDJSON (SSE) | Voice-friendly completeness over minimum latency |
| Multi-surface | Same LangGraph processor for REST + A2A | Zero logic duplication across protocols |
| Session store | In-memory with TTL | Zero deps for demo; swap to Redis with one env flag |
| Extensibility | Processor registry (Factory + Strategy) | Open/Closed Principle — add backends without touching callers |

---

## Quick Start

### Requirements

- Python 3.11 or higher
- A Gemini API key and/or an OpenAI API key

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
| `GET` | `/agent/a2a` | Agent card — capability descriptor for peer agent discovery |
| `POST` | `/agent/a2a` | A2A JSON-RPC 2.0 dispatcher (`SendMessage`, `GetTask`, `CancelTask`) |
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

The model catalog is a static Python dict in `app/agent/config/provider_catalog.py`. The env only selects which provider to prefer — no code changes needed to switch models.

### Model catalog (`provider_catalog.py`)

| Key | Provider | Model |
|---|---|---|
| `GEMINI_31_FLASH_LITE` | Google GenAI | `gemini-3.1-flash-lite` *(default — thinking model)* |
| `GPT4O_MINI` | OpenAI | `gpt-4o-mini` |
| `GEMINI_FLASH` | Google GenAI | `gemini-flash-latest` |
| `GEMINI_25_FLASH` | Google GenAI | `gemini-2.5-flash` |

To add a model: add one line to `llm_provider_map` in `provider_catalog.py`. To activate it: set `SELECTED_LLM_PROVIDER` to its key.

### Provider selection (`.env`)

```dotenv
# Set preferred provider — placed first in the failover chain:
SELECTED_LLM_PROVIDER=GEMINI_31_FLASH_LITE
```

### Runtime failover

The full provider chain is built automatically at request time: preferred first, then all other registered providers in map order. On any LLM error (rate limit, auth failure, API error), `LangGraphProcessor` retries with the next provider and emits a `FAILOVER` SSE chunk. The Streamlit chat tab shows a yellow **⚡ Failover** badge when this occurs.

To demo: set `GEMINI_API_KEY=invalid` in `.env`, restart, send a message — the gateway switches to the next provider and the badge appears.

---

## Dynamic Tool Binding

Tools are bound to the LLM **at request time** inside `agent_node` — not when the graph is compiled at startup. The `StateGraph` topology is fixed; only the tool set varies per request.

**Current behaviour:** the four built-in tools (`calculate`, `get_weather`, `search_knowledge_base`, `summarize_text`) from `ALL_TOOLS` are bound on every call.

**MCP extension point:** `GraphFlowState` carries an `llm_tools` field specifically for per-request tool injection. An MCP-aware implementation fetches the live tool manifest from MCP servers at request time, wraps each as a LangChain `@tool`, and passes them through state — `agent_node` then calls `llm.bind_tools(state["llm_tools"])` with the live set. No graph recompilation is needed; the extension is entirely local to `agent_node` in `resilient_agent_builder.py`.

---

## Project Structure

```
app/
├── agent/
│   ├── config/
│   │   └── provider_catalog.py     # Master LLM provider map (llm_provider_map)
│   ├── graph/
│   │   ├── state.py                # GraphFlowState TypedDict schema
│   │   ├── base_builder.py         # ABC for agent builders
│   │   ├── resilient_agent_builder.py      # LangGraph StateGraph + dynamic tool binding
│   │   └── middleware.py           # ResilientAgentMiddleware (model resolution + tool hooks)
│   ├── registry/
│   │   └── llm_registry.py         # Multi-provider registry + failover chain
│   ├── runner/
│   │   ├── agent_runner.py         # AgentRunner: invoke() + stream()
│   │   └── runner_manager.py       # Singleton lifecycle manager
│   ├── streaming/
│   │   ├── chunk.py                # _extract_text, _ndjson, sentence boundary regex
│   │   └── stream_processor.py     # NDJSON pipeline + sentence buffer (BUFFERED / RAW)
│   └── processors/
│       ├── base_processor.py       # BaseProcessor ABC
│       ├── langgraph_processor.py  # LangGraph backend (default, failover-aware)
│       ├── google_adk_processor.py # Google ADK stub (extension point)
│       └── processor_factory.py    # Strategy factory: AGENT_PROCESSOR env key
├── a2a/
│   ├── server.py                   # JSON-RPC 2.0 router (a2a-sdk 1.0.3)
│   ├── executor.py                 # A2A task → LangGraph stream bridge
│   ├── context_store.py            # context_id → session_id binding (multi-turn)
│   ├── task_store.py               # In-memory task state (Redis swap path)
│   └── ui_helpers.py               # Pure result-extraction helpers (testable, no Streamlit)
├── api/
│   ├── v1/
│   │   ├── chat_router.py          # POST /v1/stream, /v1/chat, /v1/session/init
│   │   └── health_router.py        # GET /health, /metrics
│   ├── middleware/
│   │   ├── logging_middleware.py   # Structured request/response logging
│   │   └── pii_filter.py           # PII scrubbing before log write
│   └── router.py                   # Root FastAPI app + route registration
├── repositories/
│   ├── base/
│   │   ├── session_repository.py   # ISessionRepository ABC
│   │   └── context_repository.py  # IContextRepository ABC
│   ├── memory/
│   │   ├── session_repository.py   # In-memory SessionStore (TTL via time.time())
│   │   └── context_repository.py  # In-memory ContextStore
│   └── redis/
│       ├── session_repository.py   # Redis-backed session store
│       ├── context_repository.py   # Redis-backed context store
│       └── task_repository.py      # Redis-backed task store
├── tools/
│   ├── tool_registry.py            # ALL_TOOLS list — single source of truth
│   ├── calculate_tool.py           # Safe AST expression evaluator
│   ├── weather_tool.py             # Mock weather data (5 cities)
│   ├── knowledge_base_tool.py      # Mock knowledge base search
│   └── summarize_tool.py           # First-two-sentences extractor
├── services/
│   └── chat_service.py             # ChatService: load_history + save_turn
├── config/
│   ├── settings.py                 # Pydantic BaseSettings
│   ├── logging_config.py           # Rotating hourly log setup
│   └── metrics.py                  # In-process metrics counters
├── models/
│   └── schemas.py                  # Pydantic request/response schemas
└── ui/
    └── streamlit_app.py            # Chat, A2A Inspector, Monitoring tabs
tests/
└── unit/                           # 64 unit tests — stream processor, registry, session, A2A
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
| Agent orchestration | LangGraph 1.2, LangChain 1.3 |
| API backend | FastAPI 0.116, Uvicorn 0.47 |
| LLM (default) | Google `gemini-3.1-flash-lite` (thinking model) |
| LLM (failover) | OpenAI `gpt-4o-mini`, `gemini-flash-latest`, `gemini-2.5-flash` |
| A2A protocol | `a2a-sdk 1.0.3` (JSON-RPC 2.0, proto-based) |
| Frontend | Streamlit 1.57 |
| Config | Pydantic v2 BaseSettings |
| Observability | contextvars trace IDs, in-process metrics, structured logging |
| Python | 3.11+ |
