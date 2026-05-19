"""
AgentRunner + AgentRunnerManager

AgentRunnerManager.initialize() — called once at startup, builds the full stack.
AgentRunnerManager.get_runner()  — returns the singleton runner everywhere else.

See docs/architecture.md for lifecycle and design rationale.
"""
import logging
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage

from app.agent.base_agent_builder import BaseAgentBuilder
from app.agent.graph_builder import GraphFlowState
from app.config.logger import get_trace_id
from app.models.schemas import RequestContext
from app.tools.assistant_tools import ALL_TOOLS

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
        state["model"] so ResilientAgentMiddleware resolves the right LLM
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


# ── Singleton manager ─────────────────────────────────────────────────────────

class AgentRunnerManager:
    """
    Singleton lifecycle manager for AgentRunner.

    Call initialize() once at application startup.
    Call get_runner() everywhere else.

    Mirrors LanggraphAgentRunnerManager from the parent project.
    """

    _runner: AgentRunner | None = None

    @classmethod
    def initialize(cls) -> None:
        """
        Wire the full agent stack and store the singleton runner.

        Execution order:
          1. Fetch populated LLMRegistry (providers already registered)
          2. Create ResilientAgentMiddleware (model resolver + tool hooks)
          3. Create ResilientAgentBuilder  (compiles StateGraph, logs ASCII graph)
          4. Wrap in AgentRunner and store
        """
        from app.agent.llm_registry import get_llm_registry
        from app.agent.middleware.agent_middleware import ResilientAgentMiddleware
        from app.agent.resilient_agent_builder import ResilientAgentBuilder

        logger.info("AgentRunnerManager: initialising agent stack…")
        registry = get_llm_registry()
        middleware = ResilientAgentMiddleware(registry)
        builder = ResilientAgentBuilder(registry, middleware)
        cls._runner = AgentRunner(builder)
        logger.info("AgentRunnerManager: ready")

    @classmethod
    def get_runner(cls) -> AgentRunner:
        if cls._runner is None:
            raise RuntimeError(
                "AgentRunnerManager is not initialised. "
                "Call AgentRunnerManager.initialize() in the application lifespan."
            )
        return cls._runner
