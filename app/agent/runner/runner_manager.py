"""
AgentRunnerManager — singleton lifecycle manager for AgentRunner.

Call initialize() once at application startup.
Call get_runner() everywhere else.

Mirrors LanggraphAgentRunnerManager from the parent project.
"""
import logging

from app.agent.runner.agent_runner import AgentRunner

logger = logging.getLogger(__name__)


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
          2. Create GatewayAgentMiddleware (model resolver + tool hooks)
          3. Create GatewayAgentBuilder  (compiles StateGraph, logs ASCII graph)
          4. Wrap in AgentRunner and store
        """
        from app.agent.registry.llm_registry import get_llm_registry
        from app.agent.graph.middleware import GatewayAgentMiddleware
        from app.agent.graph.gateway_builder import GatewayAgentBuilder

        logger.info("AgentRunnerManager: initialising agent stack…")
        registry = get_llm_registry()
        middleware = GatewayAgentMiddleware(registry)
        builder = GatewayAgentBuilder(registry, middleware)
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
