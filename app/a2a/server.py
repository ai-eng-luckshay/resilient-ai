"""
A2A JSON-RPC 2.0 server surface.

Exposes two endpoints under /agent/a2a:
  POST /agent/a2a/tasks/send    — submit a new task
  GET  /agent/a2a/tasks/{id}   — poll task state + artifacts
  GET  /agent/a2a/agent-card   — agent capability descriptor

The A2A protocol uses JSON-RPC 2.0 semantics.  We implement a lightweight
version here without the full a2a-sdk to keep the dependency footprint small
and make the protocol mechanics visible in code.

Full a2a-sdk integration is straightforward — replace the manual router with:
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandlerV2
    app.mount("/agent/a2a", A2AStarletteApplication(handler, ...).build())
"""
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.a2a.executor import A2AExecutor
from app.a2a.task_store import get_task_store
from app.config.logger import get_trace_id
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent/a2a", tags=["a2a"])
_executor = A2AExecutor()


# ── Request / Response models ──────────────────────────────────────────────

class A2AMessage(BaseModel):
    role: str = "user"
    parts: list[dict]  # [{"type": "text", "text": "..."}]


class SendTaskRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int = 1
    method: str = "tasks/send"
    params: dict  # {contextId, message: A2AMessage}


class SendTaskResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int
    result: dict


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/tasks/send")
async def send_task(body: SendTaskRequest, background_tasks: BackgroundTasks) -> SendTaskResponse:
    """
    Submit a task to the agent.

    JSON-RPC 2.0 body:
        {
          "jsonrpc": "2.0",
          "id": 1,
          "method": "tasks/send",
          "params": {
            "contextId": "<uuid>",
            "message": {"role": "user", "parts": [{"type": "text", "text": "Hello"}]}
          }
        }
    """
    params = body.params
    context_id = params.get("contextId") or str(uuid.uuid4())
    message_obj = params.get("message", {})
    parts = message_obj.get("parts", [])
    text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()

    if not text:
        raise HTTPException(status_code=400, detail="No text content in message parts.")

    settings = get_settings()
    model = settings.selected_llm_provider
    system_prompt = params.get("systemPrompt", "You are a helpful AI assistant.")
    trace_id = params.get("traceId", get_trace_id())

    # Execute asynchronously so the HTTP response returns immediately
    # with the task_id — client then polls /tasks/{id}
    task_id = await _executor.execute(
        context_id=context_id,
        message=text,
        trace_id=trace_id,
        model=model,
        system_prompt=system_prompt,
    )

    task_store = get_task_store()
    task_dict = task_store.to_dict(task_id) or {"id": task_id, "status": {"state": "working"}}

    logger.info("[%s] A2A tasks/send | context_id=%s task_id=%s", trace_id, context_id, task_id)

    return SendTaskResponse(
        id=body.id,
        result=task_dict,
    )


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    """Poll task state and artifacts."""
    task_store = get_task_store()
    task_dict = task_store.to_dict(task_id)
    if task_dict is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return task_dict


@router.get("/agent-card")
async def agent_card() -> dict:
    """Agent capability descriptor — allows peer agents to discover this gateway."""
    settings = get_settings()
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": (
            "Resilient AI — LangGraph-based agent with dynamic tool binding, "
            "LLM failover, and A2A protocol support."
        ),
        "capabilities": {
            "streaming": True,
            "a2a": True,
            "tools": ["calculate", "search_knowledge_base", "get_weather", "summarize_text"],
            "processors": ["LANGGRAPH", "GOOGLE_ADK (stub)", "A2A (stub)"],
        },
        "endpoints": {
            "tasks/send": "POST /agent/a2a/tasks/send",
            "tasks/get": "GET /agent/a2a/tasks/{task_id}",
        },
    }
