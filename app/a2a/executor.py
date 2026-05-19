"""
A2AExecutor: bridges A2A task lifecycle to the LangGraph stream pipeline.

Flow:
    1. Receive A2A task (context_id, message)
    2. Resolve or create session via ContextStore
    3. Load session history from ChatService
    4. Stream through LangGraphProcessor
    5. Append each TEXT chunk as an A2A artifact
    6. Mark task COMPLETE or FAILED

This is where the dual-surface elegance lives: the exact same processor
used by the REST /stream endpoint drives the A2A task — zero duplication.
"""
import json
import logging

from app.a2a.context_store import get_context_store
from app.a2a.task_store import A2ATask, get_task_store
from app.agent.processor_factory import get_processor
from app.config.logger import get_trace_id, set_trace_id
from app.config.metrics import get_metrics
from app.config.settings import get_settings
from app.models.schemas import ChatRequest
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)


class A2AExecutor:
    def __init__(self) -> None:
        self._context_store = get_context_store()
        self._task_store = get_task_store()
        self._chat_service = ChatService()

    async def execute(
        self,
        context_id: str,
        message: str,
        trace_id: str | None = None,
        model: str | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> str:
        """
        Execute an A2A task.  Returns the task_id so the caller can poll
        for artifacts.
        """
        if trace_id:
            set_trace_id(trace_id)

        settings = get_settings()
        if model is None:
            model = settings.selected_llm_provider

        metrics = get_metrics()
        metrics.record_a2a_task()

        # Step 1: Resolve session (reuse if context already has one)
        session_id = self._context_store.resolve_session(
            context_id,
            lambda: self._chat_service.create_session(system_prompt),
        )
        logger.info(
            "[%s] A2AExecutor | context_id=%s session_id=%s",
            get_trace_id(), context_id, session_id,
        )

        # Step 2: Create task record
        task = self._task_store.create(context_id=context_id, session_id=session_id)
        logger.info("[%s] Task created: %s", get_trace_id(), task.task_id)

        # Step 3: Stream response and collect artifacts
        history = self._chat_service.load_history(session_id)
        request = ChatRequest(
            session_id=session_id,
            message=message,
            model=model,
            processor="LANGGRAPH",
            system_prompt=system_prompt,
        )
        processor = get_processor("LANGGRAPH")

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
                    self._task_store.add_artifact(task.task_id, content)
                    logger.debug("[%s] a2a_artifact | %s", get_trace_id(), content[:60])
                elif chunk_type == "ERROR":
                    raise RuntimeError(content)
                elif chunk_type == "STATUS" and content == "COMPLETE":
                    break

            self._task_store.complete(task.task_id)
            full_reply = " ".join(text_parts).strip()
            self._chat_service.save_turn(session_id, message, full_reply)
            logger.info("[%s] Task completed: %s", get_trace_id(), task.task_id)

        except Exception as exc:
            logger.error("[%s] Task failed: %s | %s", get_trace_id(), task.task_id, exc)
            self._task_store.fail(task.task_id, str(exc))

        return task.task_id
