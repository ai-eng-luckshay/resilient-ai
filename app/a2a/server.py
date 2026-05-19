"""
A2A JSON-RPC 2.0 server — powered by a2a-sdk 1.0.3.

Mounted at /agent/a2a via app.include_router(a2a_router) in main.py.

  GET  /agent/a2a        — agent card (peer discovery endpoint)
  GET  /agent/a2a/       — agent card (alias)
  POST /agent/a2a        — JSON-RPC 2.0 dispatcher
  POST /agent/a2a/       — JSON-RPC 2.0 dispatcher (alias)

Supported JSON-RPC methods (handled by DefaultRequestHandlerV2):
  message/send           — submit a task; runs ResilientAgentExecutor
  tasks/get              — retrieve a task by ID (poll for results)
  tasks/cancel           — cancel a running task

JSON-RPC request envelope:
    {
      "jsonrpc": "2.0",
      "id": 1,
      "method": "message/send",
      "params": {
        "message": {
          "messageId": "<uuid>",
          "role": "user",
          "parts": [{"text": "Hello"}]
        },
        "contextId": "<optional-uuid-for-multi-turn>"
      }
    }
"""
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from google.protobuf.json_format import ParseDict

from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes.common import DefaultServerCallContextBuilder
from a2a.server.routes.jsonrpc_dispatcher import JsonRpcDispatcher
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard

from app.a2a.executor import ResilientAgentExecutor
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent/a2a", tags=["a2a"])

# Module-level lazy singletons — initialised on first request
_dispatcher: JsonRpcDispatcher | None = None


def _build_agent_card_dict() -> dict:
    """Build the A2A agent card as a plain dict (served as JSON and used to seed the proto)."""
    settings = get_settings()
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": (
            "Resilient AI — LangGraph-based agent gateway with dynamic tool binding, "
            "automatic LLM failover across providers, streaming sentence buffering, "
            "and A2A protocol support."
        ),
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "resilient_chat",
                "name": "Resilient Chat",
                "description": (
                    "General-purpose conversational AI with tool use (calculate, search, "
                    "weather, summarize), automatic LLM failover, and session memory. "
                    "Invoke for any question, calculation, knowledge retrieval, or text "
                    "summarization task."
                ),
                "tags": ["chat", "tools", "failover", "streaming", "multi-turn"],
                "examples": [
                    "What is 1234 * 5678?",
                    "Search for information about LangGraph",
                    "Summarize this paragraph for me",
                    "What's the weather like today?",
                    "Explain the difference between RAG and fine-tuning",
                ],
            }
        ],
    }


def _get_dispatcher() -> JsonRpcDispatcher:
    """Lazily build and cache the JSON-RPC dispatcher singleton."""
    global _dispatcher
    if _dispatcher is None:
        agent_card_proto = ParseDict(_build_agent_card_dict(), AgentCard())
        handler = DefaultRequestHandlerV2(
            agent_executor=ResilientAgentExecutor(),
            task_store=InMemoryTaskStore(),
            agent_card=agent_card_proto,
        )
        _dispatcher = JsonRpcDispatcher(
            request_handler=handler,
            context_builder=DefaultServerCallContextBuilder(),
        )
        logger.info("A2A JsonRpcDispatcher initialised (a2a-sdk 1.0.3)")
    return _dispatcher


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get("")
@router.get("/")
async def agent_card() -> JSONResponse:
    """
    Agent capability descriptor for peer agent discovery.
    Returns the static agent card so other agents can understand this gateway's skills.
    """
    return JSONResponse(
        content=_build_agent_card_dict(),
        headers={"Access-Control-Allow-Origin": "*"},
    )


@router.post("")
@router.post("/")
async def jsonrpc_endpoint(request: Request) -> Response:
    """
    JSON-RPC 2.0 dispatcher.

    Delegates to DefaultRequestHandlerV2 which routes message/send,
    tasks/get, and tasks/cancel to the appropriate handler.
    """
    return await _get_dispatcher().handle_requests(request)
