# Architecture & Design Decisions

## LLM Provider Registry

The master model catalog lives in code (`app/agent/agent_util.py`). The environment only selects which subset is active.

### Model catalog (`agent_util.py`)

```python
llm_provider_map = {
    "GEMINI_31_FLASH_LITE": "GG_gemini-3.1-flash-lite-preview",
    "GPT4O_MINI":           "O_gpt-4o-mini",
    "GEMINI_FLASH":         "GG_gemini-flash-latest",
    "GEMINI_25_FLASH":      "GG_gemini-2.5-flash",
}
```

Value format: `PREFIX_model-name`. The prefix encodes the client SDK:

| Prefix | Client SDK | Required secret |
|---|---|---|
| `O` | `ChatOpenAI` | `OPENAI_API_KEY` |
| `GG` | `ChatGoogleGenerativeAI` | `GEMINI_API_KEY` |

To add a model: add one line to `llm_provider_map`. To activate it: include its key in `LLM_PROVIDER_CHAIN` in `.env`.

### `.env` (preferred provider only)

```dotenv
SELECTED_LLM_PROVIDER=GPT4O_MINI
```

### Chain construction (dynamic)

The full failover chain is built at request time — no `LLM_PROVIDER_CHAIN` variable needed:

```
1. Start with all keys from llm_provider_map that have a registered client
   (i.e., the required API key is present), in map definition order.
2. Resolve the preferred provider:
     a. request.model   (from UI dropdown or API caller)  — if registered
     b. SELECTED_LLM_PROVIDER from .env                   — if registered
     c. first registered provider in map order            — fallback
3. Chain = [preferred] + [remaining in map order]
```

On any LLM error, `LangGraphProcessor` catches the exception, emits a `FAILOVER` SSE chunk, and retries with the next provider. If all fail, an `ERROR` chunk is returned.

---

## Agent Builder Pattern

The agent stack is structured as a builder hierarchy:

```
BaseAgentBuilder (ABC)
    └── GatewayAgentBuilder
            ├── GatewayAgentMiddleware   ← injected
            ├── LLMRegistry              ← injected
            └── StateGraph (compiled once at startup)
```

### Why compile once

In the original design, a separate graph was compiled per provider key (`_get_compiled_graph(provider_key)`). This was wasteful — the graph topology is identical for every provider. With the builder pattern, the graph is compiled once and the model is resolved per-request inside `agent_node` via `middleware.resolve_model(state["model"])`. Failover still works: `LangGraphProcessor` writes a different `provider_key` into state on each retry.

### AgentRunnerManager lifecycle

```
app startup
  └── AgentRunnerManager.initialize()
          ├── get_llm_registry()           (already populated)
          ├── GatewayAgentMiddleware(registry)
          ├── GatewayAgentBuilder(registry, middleware)
          │       └── _create_workflow()   (compiles StateGraph, logs ASCII graph)
          └── AgentRunner(builder)         → stored as singleton

request
  └── AgentRunnerManager.get_runner().stream(...)
```

### GatewayAgentMiddleware

Injected into the workflow's `agent_node` and `tool_node` closures. Provides:

- `resolve_model(state)` — reads `state["model"]` (the provider key) and returns the live `BaseChatModel` from the registry. Falls back to the first available provider on a missing key.
- `pre_tool_call(tool_name, args, context)` — logs tool entry + args, increments metrics counter, returns start timestamp.
- `post_tool_call(tool_name, result, start, context)` — logs tool exit + elapsed time.

---

## Session Store — In-Memory vs Redis

Three stores ship with in-memory implementations for the demo:

| Store | Module | Purpose |
|---|---|---|
| `SessionStore` | `app/services/session_store.py` | Chat history + system prompt per session |
| `ContextStore` | `app/a2a/context_store.py` | A2A `context_id → session_id` binding |
| `InMemoryTaskStore` | `app/a2a/task_store.py` | A2A task state + artifacts |

### Why in-memory is fine for a single process

Zero external dependencies. O(1) access. TTL managed manually via `time.time()` checks. Sufficient for demo and single-worker deployments.

### Why Redis is required at scale

| Concern | In-memory | Redis |
|---|---|---|
| Latency | ~0 µs (same process) | ~0.5–2 ms (network hop) |
| Horizontal scaling | State is per-worker — Worker B has no knowledge of sessions Worker A created | Shared store across all workers and pods |
| Durability | Lost on process restart or rolling deploy | Survives restarts; configurable persistence (RDB / AOF) |
| TTL management | Manual `time.time()` checks | Native key expiry (`EX` flag on `SET`) |
| A2A continuity | Context bindings lost on restart — polling client gets 404 for tasks that existed before restart | Bindings survive restarts and worker reassignment |

### Swap path

Set `USE_REDIS=true` and `REDIS_URL=redis://...` in `.env`. Each store factory (`get_session_store`, `get_context_store`, `get_task_store`) checks this flag and returns the Redis-backed class from `app/infra/redis_store.py`. No other code changes.

To activate, install `redis>=5.0` (add to `requirements.in`, run `pip-compile` + `pip-sync`) and remove the `NotImplementedError` guard in `app/infra/redis_store.py::_get_redis_client`.

### Redis key schema

| Store | Key pattern | TTL |
|---|---|---|
| SessionStore | `session:{session_id}` | `SESSION_TTL_SECONDS` (default 7200s) |
| ContextStore | `a2a:context:{context_id}` | `A2A_CONTEXT_TTL_SECONDS` (default 86400s) |
| TaskStore | `a2a:task:{task_id}` | 3600s |

---

## Streaming — BUFFERED vs RAW

Controlled by `STREAM_MODE` in `.env`.

| Mode | Behaviour | Latency | Best for |
|---|---|---|---|
| `BUFFERED` (default) | Accumulates tokens until a sentence boundary (`[.!?;\n—,]` followed by whitespace), then yields a complete sentence | ~1 sentence of additional latency | Voice / IVR consumers that need grammatically complete phrases |
| `RAW` | Yields every token immediately as it arrives from the LLM | Minimum | Chat UI clients |

---

## Processor Strategy Pattern

The processing backend is selectable via `AGENT_PROCESSOR` in `.env`.

```
ProcessorFactory
  ├── LANGGRAPH  → LangGraphProcessor   (fully functional, default)
  └── GOOGLE_ADK → GoogleADKProcessor   (extension stub)
```

Adding a new backend: subclass `BaseProcessor`, implement `stream()`, register in `processor_factory.py`. Zero other changes — callers use the factory and are unaware of the concrete type.

The A2A surface (`/agent/a2a/...`) uses `LangGraphProcessor` directly — same workflow, different protocol layer. This is the zero-duplication point: one compiled LangGraph graph serves both REST and A2A.
