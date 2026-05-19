# Architecture & Design Decisions

## LLM Provider Registry

The master model catalog lives in code (`app/agent/config/provider_catalog.py`). The environment only selects which subset is active.

### Model catalog (`provider_catalog.py`)

```python
llm_provider_map = {
    "GEMINI_31_FLASH_LITE": "GG_gemini-3.1-flash-lite",  # default — thinking model; langchain-google-genai 3.x preserves thought_signatures
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

To add a model: add one line to `llm_provider_map` in `provider_catalog.py`. To activate it: set `SELECTED_LLM_PROVIDER` to its key in `.env`.

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

## Dynamic Tool Binding & MCP Extension

Tools are bound to the LLM **inside `agent_node`, at request time** — not when the graph is compiled at startup:

```python
# app/agent/graph/resilient_agent_builder.py — runs per-request
async def agent_node(state: GraphFlowState) -> dict:
    llm = middleware.resolve_model(state)      # model resolved per-request
    llm_with_tools = llm.bind_tools(tools)    # tools bound per-request
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}
```

The `StateGraph` topology is compiled once at startup and never changes. The tool set is the only variable — a fresh binding on every call.

**Current behaviour:** `agent_node` uses the static `ALL_TOOLS` list from `app/tools/tool_registry.py`. The four built-in tools (`calculate`, `get_weather`, `search_knowledge_base`, `summarize_text`) are bound on every request.

**MCP extension point:** `GraphFlowState` reserves an `llm_tools` field for per-request tool injection:

```python
class GraphFlowState(TypedDict):
    ...
    llm_tools: list[Any]   # injection point — replace ALL_TOOLS with MCP-fetched tools
```

MCP integration is designed to plug in here. An MCP-aware implementation fetches the live tool manifest from one or more MCP servers at request time (or from a per-session cache), wraps each entry as a LangChain `@tool`, and populates `state["llm_tools"]`. `agent_node` then calls `llm.bind_tools(state["llm_tools"])` with the live set instead of the static list. Because binding happens inside the node — not at compile time — **zero graph recompilation is required**. The change is entirely local to `agent_node` in `app/agent/graph/resilient_agent_builder.py`.

---

## Agent Builder Pattern

The agent stack is structured as a builder hierarchy:

```
app/agent/graph/base_builder.py    → BaseAgentBuilder (ABC)
app/agent/graph/resilient_agent_builder.py → ResilientAgentBuilder
    ├── ResilientAgentMiddleware  (app/agent/graph/middleware.py)  ← injected
    ├── LLMRegistry             (app/agent/registry/llm_registry.py) ← injected
    └── StateGraph (compiled once at startup)
```

### Why compile once

In the original design, a separate graph was compiled per provider key (`_get_compiled_graph(provider_key)`). This was wasteful — the graph topology is identical for every provider. With the builder pattern, the graph is compiled once and the model is resolved per-request inside `agent_node` via `middleware.resolve_model(state["model"])`. Failover still works: `LangGraphProcessor` writes a different `provider_key` into state on each retry.

### AgentRunnerManager lifecycle

```
app startup
  └── AgentRunnerManager.initialize()          (app/agent/runner/runner_manager.py)
          ├── get_llm_registry()               (app/agent/registry/llm_registry.py)
          ├── ResilientAgentMiddleware(registry) (app/agent/graph/middleware.py)
          ├── ResilientAgentBuilder(registry, middleware)
          │       └── _create_workflow()       (compiles StateGraph, logs ASCII graph)
          └── AgentRunner(builder)             (app/agent/runner/agent_runner.py)
                  → stored as singleton

request
  └── AgentRunnerManager.get_runner().stream(...)
```

### ResilientAgentMiddleware

Injected into the workflow's `agent_node` and `tool_node` closures. Provides:

- `resolve_model(state)` — reads `state["model"]` (the provider key) and returns the live `BaseChatModel` from the registry. Falls back to the first available provider on a missing key.
- `pre_tool_call(tool_name, args, context)` — logs tool entry + args, increments metrics counter, returns start timestamp.
- `post_tool_call(tool_name, result, start, context)` — logs tool exit + elapsed time.

---

## Session Store — In-Memory vs Redis

Three stores ship with in-memory implementations for the demo:

| Store | Module | Purpose |
|---|---|---|
| `SessionStore` | `app/repositories/memory/session_repository.py` | Chat history + system prompt per session |
| `ContextStore` | `app/repositories/memory/context_repository.py` | A2A `context_id → session_id` binding |
| `InMemoryTaskStore` | `app/a2a/task_store.py` | A2A task state + artifacts |

Each store implements an ABC from `app/repositories/base/` — swapping in the Redis variant requires only changing the factory return value.

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

Set `USE_REDIS=true` and `REDIS_URL=redis://...` in `.env`. Each store factory (`get_session_store`, `get_context_store`, `get_task_store`) checks this flag and returns the Redis-backed class from `app/repositories/redis/`. No other code changes.

To activate, install `redis>=5.0` (add to `requirements.in`, run `pip-compile` + `pip-sync`). The Redis implementations live in:

| Store | Redis module |
|---|---|
| SessionStore | `app/repositories/redis/session_repository.py` |
| ContextStore | `app/repositories/redis/context_repository.py` |
| TaskStore | `app/repositories/redis/task_repository.py` |

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

### Observability difference: streaming vs non-streaming

The two chat endpoints (`POST /v1/stream` and `POST /v1/chat`) use the same `LangGraphProcessor` internally, but they differ significantly in what is surfaced to the caller.

**Failover events**

| Endpoint | How failover is visible |
|---|---|
| `POST /v1/stream` | A `FAILOVER` SSE chunk is emitted in the event stream the moment the active provider is switched. The Streamlit Chat tab renders this as a yellow **⚡ Failover: Switched to \<provider\>** badge inline in the conversation. |
| `POST /v1/chat` | The final JSON response contains only the successful reply — no indication that a failover occurred. To confirm whether failover happened, inspect the structured logs (`logs/…/gateway_SYSTEM_*.log`) and look for `FAILOVER:` entries tagged with the request's `trace_id`. |

**Tool call results**

| Endpoint | How tool calls are visible |
|---|---|
| `POST /v1/stream` | A `TOOL_RESULT` SSE chunk is emitted for every tool invocation as it completes. The Streamlit Chat tab renders each result as an expandable **🛠 Tool Calls** card in the right-hand column, showing the tool name and output. |
| `POST /v1/chat` | Only the final assistant reply is returned. Tool calls happen internally but their inputs and outputs are not included in the response body. To inspect tool activity, check the structured logs (`gateway_SYSTEM_*.log`) for `→ ToolNode` / `← ToolNode` entries, or the Monitoring tab's **Tool Calls by Tool** chart. |

This is an intentional design tradeoff: the non-streaming endpoint is simpler for integrations that only need the final answer; the streaming endpoint is richer for interactive clients and observability.

---

## A2A Protocol — `a2a-sdk 1.0.3`

The A2A surface (`app/a2a/`) uses the official **`a2a-sdk 1.0.3`** package. The SDK handles JSON-RPC 2.0 dispatching, task lifecycle management, and the event queue — the project code implements only the agent-specific logic.

### Endpoint layout

| Method | Path | Description |
|---|---|---|
| `GET` | `/agent/a2a` | Agent card — capability descriptor for peer agent discovery |
| `POST` | `/agent/a2a` | JSON-RPC 2.0 dispatcher (`SendMessage`, `GetTask`, `CancelTask`) |

### Component roles

| File | Role |
|---|---|
| `app/a2a/server.py` | Builds `JsonRpcDispatcher` + `DefaultRequestHandlerV2`; serves agent card and POST dispatcher |
| `app/a2a/executor.py` | `ResilientAgentExecutor(AgentExecutor)` — bridges A2A tasks to `LangGraphProcessor` |
| `app/a2a/context_store.py` | `context_id → session_id` binding — survives multi-turn A2A conversations |
| `app/a2a/task_store.py` | Custom `InMemoryTaskStore` kept for the Redis swap path; SDK's `InMemoryTaskStore` is used by default |
| `app/a2a/ui_helpers.py` | Pure result-extraction helpers (`a2a_extract_text`, `a2a_state_badge`) — no Streamlit dependency, fully unit-tested |

### SDK integration pattern

```python
# server.py — lazy singleton dispatcher
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes.jsonrpc_dispatcher import JsonRpcDispatcher
from a2a.server.tasks.in_memory_task_store import InMemoryTaskStore
from a2a.types import AgentCard
from google.protobuf.json_format import ParseDict

handler = DefaultRequestHandlerV2(
    agent_executor=ResilientAgentExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=ParseDict(agent_card_dict, AgentCard()),
)
dispatcher = JsonRpcDispatcher(
    request_handler=handler,
    context_builder=DefaultServerCallContextBuilder(),
)
```

### Executor pattern (`ResilientAgentExecutor`)

```python
class ResilientAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 1. Enqueue Task proto first — SDK invariant
        await event_queue.enqueue_event(Task(id=task_id, ...))

        updater = TaskUpdater(event_queue, task_id, context_id)
        user_message = context.get_user_input()

        # 2. Provider selection: SELECTED_LLM_PROVIDER from .env seeds the chain.
        #    LangGraphProcessor builds the full failover chain at runtime:
        #      chain = [preferred] + [remaining registered providers in map order]
        #    On any LLM error, the next provider is tried automatically.
        preferred_provider = settings.selected_llm_provider
        request = ChatRequest(model=preferred_provider, ...)

        # 3. Stream LangGraphProcessor — same pipeline as REST /stream
        async for chunk in processor.stream(request, history):
            if chunk["type"] == "TEXT":
                await updater.add_artifact(
                    parts=[Part(text=chunk["content"])],
                    artifact_id=artifact_id,
                    append=not first_chunk,
                    last_chunk=False,
                )

        await updater.complete()   # or updater.failed() on error
```

The `Task` proto **must** be enqueued before any `TaskStatusUpdateEvent`. This seeds the SDK's task store before the dispatcher processes response events.

The failover chain is identical to the REST `/stream` endpoint — the A2A surface does not need its own failover logic; it inherits it by calling the same `LangGraphProcessor.stream()`.

### JSON-RPC request format

The a2a-sdk 1.0.3 uses gRPC-style method names and requires the `A2A-Version: 1.0` request header. Without the header the SDK version validator defaults to `0.3` and rejects the request.

```
POST /agent/a2a
A2A-Version: 1.0
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "<uuid>",
      "role": "ROLE_USER",
      "contextId": "<optional-uuid-for-multi-turn>",
      "parts": [{"text": "What is 15 squared?"}]
    }
  }
}
```

To poll for results: `method: "GetTask"`, `params: {"id": "<task-id>"}`.

Reusing the same `contextId` routes each task to the same chat session, preserving conversation history across multiple A2A calls.

### Dual-surface zero-duplication point

The A2A surface and the REST surface share the same compiled LangGraph graph:

```
POST /v1/stream  ──┐
                   ├── LangGraphProcessor.stream() ── compiled StateGraph
POST /agent/a2a  ──┘   (same graph, same tools, same failover chain)
```

`ResilientAgentExecutor` calls `processor.stream()` exactly as the REST controller does — no duplicated agent logic.

---

## Processor Strategy Pattern

The processing backend is selectable via `AGENT_PROCESSOR` in `.env`.

```
ProcessorFactory
  ├── LANGGRAPH  → LangGraphProcessor   (fully functional, default)
  └── GOOGLE_ADK → GoogleADKProcessor   (extension stub)
```

Adding a new backend: subclass `BaseProcessor` (`app/agent/processors/base_processor.py`), implement `stream()`, register in `app/agent/processors/processor_factory.py`. Zero other changes — callers use the factory and are unaware of the concrete type.

The A2A surface (`/agent/a2a/...`) uses `LangGraphProcessor` directly — same workflow, different protocol layer. This is the zero-duplication point: one compiled LangGraph graph serves both REST and A2A.
