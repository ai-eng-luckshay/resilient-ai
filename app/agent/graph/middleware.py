"""
GatewayAgentMiddleware — intercepts model calls and tool calls inside the
LangGraph workflow.

Mirrors SYWAgentMiddleware from the parent project.  Because LangGraph 1.0.x
does not expose a middleware= parameter on create_react_agent, the middleware
is injected directly into the agent_node and tool_node closures inside
GatewayAgentBuilder._create_workflow().  The effect is identical — every
model resolution and every tool invocation passes through this class.

Responsibilities
────────────────
resolve_model(state)
    Reads state["model"] (the provider key set per-request) and returns
    the live BaseChatModel from the registry.  Falls back to the first
    available provider so the workflow never fails on a missing key.

pre_tool_call(tool_name, tool_args, context)
    Called before each tool invocation:
      - Logs tool name + truncated args with trace ID
      - Increments the per-tool metrics counter
      - Returns the start timestamp for elapsed-time calculation

post_tool_call(tool_name, result, start, context)
    Called after each tool invocation:
      - Logs tool name + elapsed time + truncated result
"""
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.config.logging_config import get_trace_id
from app.config.metrics import get_metrics
from app.models.schemas import RequestContext

logger = logging.getLogger(__name__)


class GatewayAgentMiddleware:
    """
    Stateless middleware injected into the compiled LangGraph workflow.

    One instance is created at startup (inside AgentRunnerManager.initialize)
    and shared across all requests — thread-safe because it holds no mutable
    per-request state.
    """

    def __init__(self, llm_registry) -> None:
        self._registry = llm_registry

    # ── Model resolution ──────────────────────────────────────────────────

    def resolve_model(self, state: dict) -> BaseChatModel:
        """
        Return the LLM for this request by reading state["model"].

        The model key is set by LangGraphProcessor before invoking the graph
        (e.g. "GPT4O_MINI" or "GEMINI_FLASH").  If the key is absent or
        unregistered the first available provider is used as a fallback,
        so the workflow is resilient to misconfigured requests.
        """
        model_key = state.get("model", "")
        try:
            return self._registry.get(model_key)
        except KeyError:
            available = self._registry.available()
            if not available:
                raise RuntimeError(
                    "No LLM providers registered. "
                    "Add LLM_O_<KEY> or LLM_GG_<KEY> entries to .env."
                )
            fallback = available[0]
            logger.warning(
                "[%s] Model key '%s' not registered — falling back to '%s'",
                get_trace_id(), model_key, fallback,
            )
            return self._registry.get(fallback)

    # ── Tool call hooks ───────────────────────────────────────────────────

    def pre_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        context: RequestContext | None,
    ) -> float:
        """
        Called immediately before a tool is invoked.

        Logs the tool name + truncated args and increments the metrics counter.
        Returns the high-resolution start time so post_tool_call can compute
        elapsed milliseconds.
        """
        start = time.perf_counter()
        trace = get_trace_id()
        logger.info(
            "[%s] → ToolNode | tool=%s | args=%r",
            trace, tool_name, str(tool_args)[:120],
        )
        get_metrics().record_tool_call(tool_name)
        return start

    def post_tool_call(
        self,
        tool_name: str,
        result: Any,
        start: float,
        context: RequestContext | None,
    ) -> None:
        """
        Called immediately after a tool returns (or raises).

        Logs the tool name, elapsed time, and a truncated result preview.
        """
        elapsed_ms = (time.perf_counter() - start) * 1000
        trace = get_trace_id()
        logger.info(
            "[%s] ← ToolNode | tool=%s | %.1fms | result=%r",
            trace, tool_name, elapsed_ms, str(result)[:80],
        )
