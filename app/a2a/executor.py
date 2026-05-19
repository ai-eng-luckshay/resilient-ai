"""
A2A AgentExecutor — bridges A2A task lifecycle to the LangGraph stream pipeline.

Implements the a2a-sdk AgentExecutor interface. The SDK's DefaultRequestHandlerV2
calls execute() for each incoming task, managing the event queue and task store
lifecycle automatically.

Flow:
    1. Enqueue Task proto first  (SDK requirement — must precede any status events)
    2. Resolve or create session via ContextStore
    3. Load history from ChatService
    4. Stream through LangGraphProcessor (same pipeline used by REST /stream)
    5. Emit TEXT chunks as A2A artifacts via TaskUpdater
    6. Call updater.complete() or updater.failed()

This is where the dual-surface elegance lives: the exact same LangGraph
processor that drives the REST /stream endpoint also drives A2A tasks —
zero duplication, one compiled graph.
"""
import json
import logging
import uuid

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from app.a2a.context_store import get_context_store
from app.agent.processor_factory import get_processor
from app.config.logger import get_trace_id, set_trace_id
from app.config.metrics import get_metrics
from app.config.settings import get_settings
from app.models.schemas import ChatRequest
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)


class ResilientAgentExecutor(AgentExecutor):
    """
    A2A AgentExecutor that streams responses through the LangGraphProcessor.

    Registered with DefaultRequestHandlerV2 in server.py. The SDK calls
    execute() for each message/send request and cancel() for cancellations.
    """

    def __init__(self) -> None:
        self._context_store = get_context_store()
        self._chat_service = ChatService()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid.uuid4())
        context_id = context.context_id or task_id

        # ── Must be first ──────────────────────────────────────────────────────
        # The SDK's DefaultRequestHandlerV2 passes task=None to RequestContext for
        # new tasks. Enqueueing the Task proto here seeds the task store before
        # any TaskStatusUpdateEvent is emitted, which is an SDK invariant.
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
        )

        updater = TaskUpdater(event_queue, task_id, context_id)

        user_message = context.get_user_input()
        if not user_message:
            logger.warning("[%s] A2A task %s — empty user input", get_trace_id(), task_id)
            await updater.failed()
            return

        # ── Provider selection ─────────────────────────────────────────────────
        # Use SELECTED_LLM_PROVIDER from .env as the preferred entry-point.
        # LangGraphProcessor.stream() builds the full failover chain at runtime:
        #   chain = [preferred] + [remaining registered providers in llm_provider_map order]
        # If the preferred provider fails, the next one in the chain is tried
        # automatically and a FAILOVER chunk is emitted (visible in server logs).
        # If no registered provider matches selected_llm_provider, the first
        # available registered provider is used instead.
        settings = get_settings()
        preferred_provider = settings.selected_llm_provider
        system_prompt = "You are a helpful AI assistant."

        # ── Session resolution ─────────────────────────────────────────────────
        # Reuse an existing session if this context_id has been seen before,
        # so multi-turn A2A conversations retain full history.
        session_id = self._context_store.resolve_session(
            context_id,
            lambda: self._chat_service.create_session(system_prompt),
        )
        logger.info(
            "[%s] A2A execute | task_id=%s context_id=%s session_id=%s preferred_provider=%s",
            get_trace_id(), task_id, context_id, session_id, preferred_provider,
        )

        get_metrics().record_a2a_task()

        history = self._chat_service.load_history(session_id)
        request = ChatRequest(
            session_id=session_id,
            message=user_message,
            model=preferred_provider,   # seeds the failover chain in LangGraphProcessor
            processor="LANGGRAPH",
            system_prompt=system_prompt,
        )
        processor = get_processor("LANGGRAPH")

        artifact_id = str(uuid.uuid4())
        first_chunk = True
        text_parts: list[str] = []

        try:
            async for raw_chunk in processor.stream(request, history):
                try:
                    chunk = json.loads(raw_chunk.strip())
                except json.JSONDecodeError:
                    continue

                chunk_type = chunk.get("type")
                content = chunk.get("content", "")

                if chunk_type == "TEXT" and content:
                    text_parts.append(content)
                    await updater.add_artifact(
                        parts=[Part(text=content)],
                        artifact_id=artifact_id,
                        append=not first_chunk,
                        last_chunk=False,
                    )
                    first_chunk = False

                elif chunk_type == "STATUS" and content == "COMPLETE":
                    await updater.complete()
                    self._chat_service.save_turn(
                        session_id, user_message, " ".join(text_parts).strip()
                    )
                    return

                elif chunk_type == "ERROR":
                    logger.error(
                        "[%s] A2A stream error | task_id=%s: %s",
                        get_trace_id(), task_id, content,
                    )
                    await updater.failed()
                    return

            # Stream ended without an explicit STATUS=COMPLETE chunk
            await updater.complete()
            self._chat_service.save_turn(
                session_id, user_message, " ".join(text_parts).strip()
            )

        except Exception as exc:
            logger.error(
                "[%s] A2A executor error | task_id=%s: %s",
                get_trace_id(), task_id, exc,
            )
            await updater.failed()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or task_id
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()
