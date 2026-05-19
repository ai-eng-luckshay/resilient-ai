"""
Unit tests for ResilientAgentExecutor.

Tests the A2A executor's chunk-routing logic — verifying that it correctly
maps LangGraphProcessor output chunks to A2A SDK TaskUpdater calls:

  - TEXT chunks     → add_artifact() called per chunk, appended after the first
  - STATUS=COMPLETE → complete() called; chat turn saved
  - ERROR chunk     → failed() called; no turn saved
  - FAILOVER chunk  → silently ignored (not forwarded as an artifact)
  - Empty input     → failed() immediately, no processor call
  - Stream ends without STATUS=COMPLETE → complete() still called (safety net)
"""
import json
from contextlib import contextmanager, ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.a2a.executor import ResilientAgentExecutor


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.task_id = "task-abc"
    ctx.context_id = "ctx-abc"
    ctx.get_user_input.return_value = "What is 15 squared?"
    return ctx


@pytest.fixture
def mock_event_queue():
    q = MagicMock()
    q.enqueue_event = AsyncMock()
    return q


def _stream(*chunks):
    """Return an async generator that yields pre-formatted NDJSON strings."""
    async def gen():
        for c in chunks:
            yield json.dumps(c) + "\n"
    return gen()


# ── Patch helper ───────────────────────────────────────────────────────────

@contextmanager
def _patch_executor_deps(processor_stream=None):
    """
    Context manager that patches all ResilientAgentExecutor dependencies and
    yields (mock_updater, mock_chat_service).

    The processor mock's stream() returns `processor_stream` if provided,
    or an empty async generator by default.
    """
    with ExitStack() as stack:
        mock_cs = stack.enter_context(patch("app.a2a.executor.get_context_store"))
        mock_cs.return_value.resolve_session.return_value = "sess-1"

        mock_chat_cls = stack.enter_context(patch("app.a2a.executor.ChatService"))
        mock_chat = mock_chat_cls.return_value
        mock_chat.load_history.return_value = []
        mock_chat.save_turn = MagicMock()

        mock_settings = stack.enter_context(patch("app.a2a.executor.get_settings"))
        mock_settings.return_value.selected_llm_provider = "GEMINI_31_FLASH_LITE"

        mock_metrics = stack.enter_context(patch("app.a2a.executor.get_metrics"))
        mock_metrics.return_value.record_a2a_task = MagicMock()

        stack.enter_context(patch("app.a2a.executor.get_trace_id", return_value="trace-1"))
        stack.enter_context(patch("app.a2a.executor.set_trace_id"))

        mock_proc_factory = stack.enter_context(patch("app.a2a.executor.get_processor"))
        mock_proc = MagicMock()
        mock_proc.stream.return_value = processor_stream if processor_stream is not None else _stream()
        mock_proc_factory.return_value = mock_proc

        mock_updater_cls = stack.enter_context(patch("app.a2a.executor.TaskUpdater"))
        mock_updater = MagicMock()
        mock_updater.add_artifact = AsyncMock()
        mock_updater.complete = AsyncMock()
        mock_updater.failed = AsyncMock()
        mock_updater.cancel = AsyncMock()
        mock_updater_cls.return_value = mock_updater

        yield mock_updater, mock_chat


# ── TEXT chunk routing ─────────────────────────────────────────────────────

class TestResilientAgentExecutorTextChunks:
    """TEXT chunks must be forwarded as TaskUpdater artifact parts."""

    @pytest.mark.asyncio
    async def test_single_text_chunk_calls_add_artifact(
        self, mock_context, mock_event_queue
    ):
        stream = _stream(
            {"type": "TEXT", "content": "15 squared is 225."},
            {"type": "STATUS", "content": "COMPLETE"},
        )
        with _patch_executor_deps(stream) as (updater, _):
            await ResilientAgentExecutor().execute(mock_context, mock_event_queue)

        updater.add_artifact.assert_awaited_once()
        part = updater.add_artifact.call_args.kwargs["parts"][0]
        assert part.text == "15 squared is 225."
        assert updater.add_artifact.call_args.kwargs["append"] is False   # first chunk

    @pytest.mark.asyncio
    async def test_multiple_text_chunks_appended(
        self, mock_context, mock_event_queue
    ):
        stream = _stream(
            {"type": "TEXT", "content": "15 squared"},
            {"type": "TEXT", "content": " is 225."},
            {"type": "STATUS", "content": "COMPLETE"},
        )
        with _patch_executor_deps(stream) as (updater, _):
            await ResilientAgentExecutor().execute(mock_context, mock_event_queue)

        assert updater.add_artifact.await_count == 2
        first_call = updater.add_artifact.call_args_list[0]
        second_call = updater.add_artifact.call_args_list[1]
        assert first_call.kwargs["append"] is False   # first chunk
        assert second_call.kwargs["append"] is True   # subsequent chunks


# ── Completion ─────────────────────────────────────────────────────────────

class TestResilientAgentExecutorCompletion:
    """STATUS=COMPLETE must call updater.complete() and save the chat turn."""

    @pytest.mark.asyncio
    async def test_status_complete_calls_updater_complete(
        self, mock_context, mock_event_queue
    ):
        stream = _stream(
            {"type": "TEXT", "content": "Hello"},
            {"type": "STATUS", "content": "COMPLETE"},
        )
        with _patch_executor_deps(stream) as (updater, chat_svc):
            await ResilientAgentExecutor().execute(mock_context, mock_event_queue)

        updater.complete.assert_awaited_once()
        updater.failed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_complete_saves_chat_turn(
        self, mock_context, mock_event_queue
    ):
        stream = _stream(
            {"type": "TEXT", "content": "15 squared"},
            {"type": "TEXT", "content": " is 225."},
            {"type": "STATUS", "content": "COMPLETE"},
        )
        with _patch_executor_deps(stream) as (updater, chat_svc):
            await ResilientAgentExecutor().execute(mock_context, mock_event_queue)

        chat_svc.save_turn.assert_called_once()
        _, user_msg, assistant_msg = chat_svc.save_turn.call_args[0]
        assert user_msg == "What is 15 squared?"
        assert "15 squared" in assistant_msg

    @pytest.mark.asyncio
    async def test_stream_ends_without_complete_chunk_still_completes(
        self, mock_context, mock_event_queue
    ):
        """Safety net: if stream ends without STATUS=COMPLETE, complete() is called."""
        stream = _stream({"type": "TEXT", "content": "Some text"})
        with _patch_executor_deps(stream) as (updater, _):
            await ResilientAgentExecutor().execute(mock_context, mock_event_queue)

        updater.complete.assert_awaited_once()


# ── Error handling ─────────────────────────────────────────────────────────

class TestResilientAgentExecutorError:
    """ERROR chunks must call updater.failed() and not save a turn."""

    @pytest.mark.asyncio
    async def test_error_chunk_calls_failed(
        self, mock_context, mock_event_queue
    ):
        stream = _stream({"type": "ERROR", "content": "All providers failed."})
        with _patch_executor_deps(stream) as (updater, chat_svc):
            await ResilientAgentExecutor().execute(mock_context, mock_event_queue)

        updater.failed.assert_awaited_once()
        updater.complete.assert_not_awaited()
        chat_svc.save_turn.assert_not_called()


# ── FAILOVER chunk filtering ───────────────────────────────────────────────

class TestResilientAgentExecutorFailover:
    """FAILOVER chunks must be silently ignored — not forwarded as artifacts."""

    @pytest.mark.asyncio
    async def test_failover_chunk_not_emitted_as_artifact(
        self, mock_context, mock_event_queue
    ):
        stream = _stream(
            {"type": "FAILOVER", "content": "Switched to GPT4O_MINI",
             "metadata": {"from": "GEMINI_31_FLASH_LITE", "to": "GPT4O_MINI"}},
            {"type": "TEXT", "content": "225"},
            {"type": "STATUS", "content": "COMPLETE"},
        )
        with _patch_executor_deps(stream) as (updater, _):
            await ResilientAgentExecutor().execute(mock_context, mock_event_queue)

        # Only the TEXT chunk should produce an artifact — not the FAILOVER
        updater.add_artifact.assert_awaited_once()
        part = updater.add_artifact.call_args.kwargs["parts"][0]
        assert part.text == "225"


# ── Empty input guard ──────────────────────────────────────────────────────

class TestResilientAgentExecutorEmptyInput:
    """Empty user input must fail immediately without calling the processor."""

    @pytest.mark.asyncio
    async def test_empty_input_calls_failed(self, mock_event_queue):
        ctx = MagicMock()
        ctx.task_id = "task-empty"
        ctx.context_id = "ctx-empty"
        ctx.get_user_input.return_value = ""

        with _patch_executor_deps() as (updater, _):
            await ResilientAgentExecutor().execute(ctx, mock_event_queue)

        updater.failed.assert_awaited_once()
        updater.add_artifact.assert_not_awaited()


# ── Cancel ─────────────────────────────────────────────────────────────────

class TestResilientAgentExecutorCancel:
    """cancel() must call updater.cancel()."""

    @pytest.mark.asyncio
    async def test_cancel_calls_updater_cancel(self, mock_event_queue):
        ctx = MagicMock()
        ctx.task_id = "task-cancel"
        ctx.context_id = "ctx-cancel"

        with patch("app.a2a.executor.TaskUpdater") as mock_updater_cls:
            mock_updater = MagicMock()
            mock_updater.cancel = AsyncMock()
            mock_updater_cls.return_value = mock_updater

            executor = ResilientAgentExecutor.__new__(ResilientAgentExecutor)
            await executor.cancel(ctx, mock_event_queue)

        mock_updater.cancel.assert_awaited_once()
