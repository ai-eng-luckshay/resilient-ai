"""
LangGraphProcessor — the default, fully-functional processor backend.

Execution flow:
  1. Build ordered provider chain (preferred first)
  2. Build GraphFlowState from session history + user message + context
  3. Attempt to stream the LangGraph workflow for each registered provider in order
  4. On execution failure, emit a FAILOVER chunk and retry with the next provider
  5. If all providers fail, emit an ERROR chunk

This is the showpiece component: it composes LLM registry, graph execution,
and streaming into a clean async generator pipeline.
"""
import json
import logging
from typing import AsyncIterator

from app.agent.agent_util import llm_provider_map
from app.agent.runner import AgentRunnerManager
from app.agent.stream_processor import StreamProcessor
from app.agent.processors.base import BaseProcessor
from app.config.logger import get_trace_id
from app.config.metrics import get_metrics
from app.config.settings import get_settings
from app.agent.llm_registry import get_llm_registry
from app.models.schemas import ChatRequest, RequestContext

logger = logging.getLogger(__name__)


class LangGraphProcessor(BaseProcessor):
    def __init__(self) -> None:
        # Runner is a singleton built at startup via AgentRunnerManager.initialize()
        self._stream_proc = StreamProcessor()

    @property
    def _runner(self):
        return AgentRunnerManager.get_runner()

    async def stream(self, request: ChatRequest, history: list) -> AsyncIterator[str]:
        settings = get_settings()
        registry = get_llm_registry()
        trace = get_trace_id()
        metrics = get_metrics()

        # --- Step 1: Build ordered provider chain dynamically ---
        # Full chain = all registered keys in llm_provider_map order.
        # Preferred = request.model (user/UI input) if registered,
        #             else SELECTED_LLM_PROVIDER from env,
        #             else first registered provider.
        registered = registry.available()
        map_ordered = [k for k in llm_provider_map if k in registered]

        preferred = (
            request.model if request.model in registered
            else settings.selected_llm_provider if settings.selected_llm_provider in registered
            else map_ordered[0] if map_ordered else None
        )

        if not preferred:
            yield json.dumps({"type": "ERROR", "content": "No LLM providers registered."}) + "\n"
            return

        providers_to_try = [preferred] + [k for k in map_ordered if k != preferred]
        # --- Step 2: Build request context ---
        context = RequestContext(
            session_id=request.session_id,
            trace_id=trace,
            system_prompt=request.system_prompt,
        )

        # --- Step 3 + 4: Stream with runtime failover ---
        # Each provider is tried in order. A FAILOVER chunk is emitted whenever the
        # active provider changes — either because the preferred was not registered,
        # or because execution failed and we are retrying with the next provider.
        last_exc: Exception | None = None
        prev_provider: str | None = None

        for provider_key in providers_to_try:
            if provider_key != preferred or prev_provider is not None:
                from_key = prev_provider or preferred
                logger.warning("[%s] FAILOVER: %s → %s", trace, from_key, provider_key)
                metrics.record_failover(from_key)
                yield (
                    json.dumps({
                        "type": "FAILOVER",
                        "content": f"Switched to {provider_key}",
                        "metadata": {"from": from_key, "to": provider_key},
                    }) + "\n"
                )

            metrics.record_llm_call(provider_key)
            logger.info(
                "[%s] LangGraphProcessor.stream | provider=%s | session=%s",
                trace, provider_key, request.session_id,
            )

            event_stream = self._runner.stream(
                history=history,
                user_message=request.message,
                provider_key=provider_key,
                context=context,
            )

            try:
                async for chunk in self._stream_proc.process(event_stream, provider_key):
                    yield chunk
                return  # provider succeeded
            except Exception as exc:
                logger.warning(
                    "[%s] Provider %s failed during execution: %s", trace, provider_key, exc
                )
                last_exc = exc
                prev_provider = provider_key

        yield json.dumps({
            "type": "ERROR",
            "content": f"All LLM providers failed. Last error: {last_exc}",
        }) + "\n"
