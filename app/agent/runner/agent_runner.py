"""
AgentRunner — thin execution wrapper around a compiled LangGraph workflow.

See docs/architecture.md for lifecycle and design rationale.
"""
import logging
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage

from app.agent.graph.base_builder import BaseAgentBuilder
from app.agent.graph.state import GraphFlowState
from app.config.logging_config import get_trace_id
from app.models.schemas import RequestContext
from app.tools.tool_registry import ALL_TOOLS

logger = logging.getLogger(__name__)


class AgentRunner:
    """
    Thin execution wrapper around a compiled LangGraph workflow.

    Accepts the compiled workflow from any BaseAgentBuilder subclass —
    decoupled from builder implementation details.
    """

    def __init__(self, builder: BaseAgentBuilder) -> None:
        self._workflow = builder.get_workflow()
        builder.draw_graph()  # logs ASCII graph at startup — visible in logs
        logger.info("AgentRunner ready")

    # ── Public API ────────────────────────────────────────────────────────

    async def stream(
        self,
        history: list,
        user_message: str,
        provider_key: str,
        context: RequestContext,
    ) -> AsyncIterator[tuple[str, Any]]:
        """
        Async generator.  Yields (mode, data) tuples from LangGraph astream:
            ("messages", (AIMessageChunk, metadata_dict))
            ("updates",  {node_name: state_delta})

        StreamProcessor handles both shapes.  provider_key is written into
        state["model"] so GatewayAgentMiddleware resolves the right LLM
        inside the workflow.
        """
        state = self._build_state(history, user_message, provider_key, context)
        trace = get_trace_id()
        logger.info(
            "[%s] AgentRunner.stream | provider=%s | session=%s",
            trace, provider_key, context.session_id,
        )
        async for event in self._workflow.astream(
            state,
            stream_mode=["messages", "updates"],
        ):
            yield event

    async def invoke(
        self,
        history: list,
        user_message: str,
        provider_key: str,
        context: RequestContext,
    ) -> GraphFlowState:
        """Blocking invocation — returns the final workflow state."""
        state = self._build_state(history, user_message, provider_key, context)
        trace = get_trace_id()
        logger.info(
            "[%s] AgentRunner.invoke | provider=%s | session=%s",
            trace, provider_key, context.session_id,
        )
        return await self._workflow.ainvoke(state)

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_state(
        history: list,
        user_message: str,
        provider_key: str,
        context: RequestContext,
    ) -> GraphFlowState:
        return {
            "messages": list(history) + [HumanMessage(content=user_message)],
            "model": provider_key,
            "request_context": context,
            "llm_tools": [t.name for t in ALL_TOOLS],
        }
