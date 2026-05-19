"""
GraphFlowState — the shared state schema threaded through the LangGraph workflow.

Kept in its own module so it can be imported by the builder, runner, tools,
and tests without pulling in graph-construction dependencies.

Graph construction lives in GatewayAgentBuilder._create_workflow().
"""
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.models.schemas import RequestContext


class GraphFlowState(TypedDict):
    # Accumulates messages across turns; add_messages handles deduplication
    messages: Annotated[list[AnyMessage], add_messages]

    # LLM provider key for this request (e.g. "GPT4O_MINI", "GEMINI_FLASH").
    # GatewayAgentMiddleware.resolve_model() reads this at node execution time
    # to return the right BaseChatModel — enabling per-request model switching
    # and failover without recompiling the graph.
    model: str

    # Per-request context threaded through the workflow — available to tools
    # via InjectedState without polluting the LLM's visible input schema.
    request_context: RequestContext

    # Names of tools available for this request (reserved for future
    # per-request tool filtering; currently carries all tool names).
    llm_tools: list[Any]
