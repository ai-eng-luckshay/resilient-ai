"""
Unit tests for StreamProcessor and _extract_text.

Tests the core streaming pipeline logic:
  - _extract_text: handles plain str and LangChain 1.x list-of-content-blocks
  - BUFFERED mode: sentence-level chunking (str and list-of-blocks content)
  - RAW mode: immediate token yield (str and list-of-blocks content)
  - LangGraph 1.x tuple event format ("messages", (chunk, metadata))
  - Signature-only blocks are silently dropped
  - Error handling: re-raises so LangGraphProcessor can trigger failover
  - STATUS:COMPLETE is always the last chunk on success
"""
import json
from typing import AsyncIterator
from unittest.mock import patch

import pytest

from app.agent.stream_processor import StreamProcessor, _ndjson, _extract_text


# ── Helpers ────────────────────────────────────────────────────────────────

async def _make_stream(*events) -> AsyncIterator:
    for e in events:
        yield e


async def _collect(processor: StreamProcessor, events) -> list[dict]:
    chunks = []
    async for raw in processor.process(_make_stream(*events), "GPT4O_MINI"):
        chunks.append(json.loads(raw.strip()))
    return chunks


# ── _ndjson helper ─────────────────────────────────────────────────────────

class TestNdjsonHelper:
    def test_basic_text_chunk(self):
        raw = _ndjson("TEXT", "Hello world")
        data = json.loads(raw.strip())
        assert data["type"] == "TEXT"
        assert data["content"] == "Hello world"

    def test_metadata_included(self):
        raw = _ndjson("TOOL_RESULT", "42", tool="calculate")
        data = json.loads(raw.strip())
        assert data["metadata"]["tool"] == "calculate"

    def test_trailing_newline(self):
        raw = _ndjson("STATUS", "COMPLETE")
        assert raw.endswith("\n")


# ── _extract_text ──────────────────────────────────────────────────────────

class TestExtractText:
    """
    _extract_text normalises the two AIMessageChunk.content formats produced
    by LangChain across versions:
      - str  (pre-1.x / non-thinking models / OpenAI)
      - list of typed content blocks (LangChain 1.x + thinking models / Gemini 3.x)
    """

    def test_plain_string_returned_as_is(self):
        assert _extract_text("Hello!") == "Hello!"

    def test_empty_string_returned_as_is(self):
        assert _extract_text("") == ""

    def test_list_single_text_block(self):
        blocks = [{"type": "text", "text": "Hello!", "index": 0}]
        assert _extract_text(blocks) == "Hello!"

    def test_list_multiple_text_blocks_concatenated(self):
        blocks = [
            {"type": "text", "text": "Hello", "index": 0},
            {"type": "text", "text": " world", "index": 0},
        ]
        assert _extract_text(blocks) == "Hello world"

    def test_signature_only_block_produces_empty_string(self):
        """Thought-signature chunks from Gemini have empty text — must be dropped."""
        blocks = [{"type": "text", "text": "", "extras": {"signature": "EjQKMg..."}, "index": 0}]
        assert _extract_text(blocks) == ""

    def test_mixed_text_and_signature_blocks(self):
        """Only text items with non-empty text are included."""
        blocks = [
            {"type": "text", "text": "15 squared is 225.", "index": 0},
            {"type": "text", "text": "", "extras": {"signature": "EjQKMg..."}, "index": 0},
        ]
        assert _extract_text(blocks) == "15 squared is 225."

    def test_empty_list_returns_empty_string(self):
        assert _extract_text([]) == ""

    def test_list_with_non_text_type_ignored(self):
        blocks = [{"type": "tool_use", "id": "abc", "name": "calculate"}]
        assert _extract_text(blocks) == ""

    def test_non_string_non_list_falls_back_to_str(self):
        """Defensive: unknown content type → str() conversion."""
        assert _extract_text(42) == "42"


# ── BUFFERED mode ──────────────────────────────────────────────────────────

class TestStreamProcessorBuffered:
    """Tests with STREAM_MODE=BUFFERED (default)."""

    @pytest.fixture
    def processor(self):
        with patch("app.agent.stream_processor.get_settings") as mock_settings:
            mock_settings.return_value.stream_mode = "BUFFERED"
            yield StreamProcessor()

    @pytest.mark.asyncio
    async def test_complete_sentence_yielded_str_content(self, processor):
        """Plain string content (OpenAI / non-thinking models)."""
        from langchain_core.messages import AIMessageChunk

        events = [
            {"messages": [AIMessageChunk(content="Hello there. ")]},
            {"messages": [AIMessageChunk(content="How are you?")]},
        ]
        chunks = await _collect(processor, events)
        types = [c["type"] for c in chunks]
        assert "TEXT" in types
        assert chunks[-1] == {"type": "STATUS", "content": "COMPLETE"}

    @pytest.mark.asyncio
    async def test_complete_sentence_yielded_list_content(self, processor):
        """List-of-blocks content (LangChain 1.x + Gemini thinking models)."""
        from langchain_core.messages import AIMessageChunk

        events = [
            {"messages": [AIMessageChunk(content=[{"type": "text", "text": "Hello there. ", "index": 0}])]},
            {"messages": [AIMessageChunk(content=[{"type": "text", "text": "How are you?", "index": 0}])]},
        ]
        chunks = await _collect(processor, events)
        text_chunks = [c for c in chunks if c["type"] == "TEXT"]
        assert len(text_chunks) >= 1
        full_text = "".join(c["content"] for c in text_chunks)
        assert "Hello there" in full_text
        assert chunks[-1] == {"type": "STATUS", "content": "COMPLETE"}

    @pytest.mark.asyncio
    async def test_signature_only_blocks_silently_dropped(self, processor):
        """Thought-signature chunks must not produce TEXT chunks."""
        from langchain_core.messages import AIMessageChunk

        events = [
            {"messages": [AIMessageChunk(content=[{"type": "text", "text": "Answer: 42.", "index": 0}])]},
            # Signature-only block — empty text, extras.signature present
            {"messages": [AIMessageChunk(content=[{"type": "text", "text": "", "extras": {"signature": "EjQKMg=="}, "index": 0}])]},
        ]
        chunks = await _collect(processor, events)
        text_chunks = [c for c in chunks if c["type"] == "TEXT"]
        # Signature block must not appear in any TEXT content
        all_content = "".join(c["content"] for c in text_chunks)
        assert "EjQKMg" not in all_content
        assert chunks[-1]["type"] == "STATUS"

    @pytest.mark.asyncio
    async def test_status_complete_always_last(self, processor):
        events = [{"messages": []}]
        chunks = await _collect(processor, events)
        assert chunks[-1]["type"] == "STATUS"
        assert chunks[-1]["content"] == "COMPLETE"

    @pytest.mark.asyncio
    async def test_exception_propagates_for_failover(self, processor):
        """
        StreamProcessor re-raises stream errors so LangGraphProcessor
        can catch them and trigger LLM provider failover.
        """
        async def bad_stream():
            yield {"messages": []}
            raise RuntimeError("test failure")

        with pytest.raises(RuntimeError, match="test failure"):
            async for _ in processor.process(bad_stream(), "GPT4O_MINI"):
                pass


# ── RAW mode ───────────────────────────────────────────────────────────────

class TestStreamProcessorRaw:
    """Tests with STREAM_MODE=RAW."""

    @pytest.fixture
    def processor(self):
        with patch("app.agent.stream_processor.get_settings") as mock_settings:
            mock_settings.return_value.stream_mode = "RAW"
            yield StreamProcessor()

    @pytest.mark.asyncio
    async def test_raw_tokens_yielded_immediately_str_content(self, processor):
        """Plain string content tokens are yielded one-per-chunk in RAW mode."""
        from langchain_core.messages import AIMessageChunk

        events = [
            {"messages": [AIMessageChunk(content="Hello")]},
            {"messages": [AIMessageChunk(content=" world")]},
        ]
        chunks = await _collect(processor, events)
        text_chunks = [c for c in chunks if c["type"] == "TEXT"]
        assert len(text_chunks) == 2
        assert text_chunks[0]["content"] == "Hello"
        assert text_chunks[1]["content"] == " world"

    @pytest.mark.asyncio
    async def test_raw_tokens_yielded_immediately_list_content(self, processor):
        """List-of-blocks tokens are extracted and yielded one-per-chunk in RAW mode."""
        from langchain_core.messages import AIMessageChunk

        events = [
            {"messages": [AIMessageChunk(content=[{"type": "text", "text": "Hello", "index": 0}])]},
            {"messages": [AIMessageChunk(content=[{"type": "text", "text": " world", "index": 0}])]},
        ]
        chunks = await _collect(processor, events)
        text_chunks = [c for c in chunks if c["type"] == "TEXT"]
        assert len(text_chunks) == 2
        assert text_chunks[0]["content"] == "Hello"
        assert text_chunks[1]["content"] == " world"

    @pytest.mark.asyncio
    async def test_signature_block_not_yielded_raw_mode(self, processor):
        """Signature-only blocks must produce no TEXT chunk even in RAW mode."""
        from langchain_core.messages import AIMessageChunk

        events = [
            {"messages": [AIMessageChunk(content=[{"type": "text", "text": "Hi", "index": 0}])]},
            {"messages": [AIMessageChunk(content=[{"type": "text", "text": "", "extras": {"signature": "abc"}, "index": 0}])]},
        ]
        chunks = await _collect(processor, events)
        text_chunks = [c for c in chunks if c["type"] == "TEXT"]
        # Only the "Hi" chunk; the empty-text signature block is discarded
        assert len(text_chunks) == 1
        assert text_chunks[0]["content"] == "Hi"


# ── LangGraph 1.x tuple event format ──────────────────────────────────────

class TestLangGraph1xTupleEvents:
    """
    LangGraph 1.0.x with stream_mode=["messages","updates"] yields
    ("messages", (AIMessageChunk, metadata_dict)) tuples.
    StreamProcessor must handle both tuple and dict event formats.
    """

    @pytest.fixture
    def processor(self):
        with patch("app.agent.stream_processor.get_settings") as mock_settings:
            mock_settings.return_value.stream_mode = "RAW"
            yield StreamProcessor()

    @pytest.mark.asyncio
    async def test_tuple_messages_event_str_content(self, processor):
        from langchain_core.messages import AIMessageChunk

        chunk = AIMessageChunk(content="Hello from tuple")
        events = [("messages", (chunk, {"langgraph_node": "agent_node"}))]
        chunks = await _collect(processor, events)
        text_chunks = [c for c in chunks if c["type"] == "TEXT"]
        assert len(text_chunks) == 1
        assert text_chunks[0]["content"] == "Hello from tuple"

    @pytest.mark.asyncio
    async def test_tuple_messages_event_list_content(self, processor):
        from langchain_core.messages import AIMessageChunk

        chunk = AIMessageChunk(content=[{"type": "text", "text": "Hello from tuple", "index": 0}])
        events = [("messages", (chunk, {"langgraph_node": "agent_node"}))]
        chunks = await _collect(processor, events)
        text_chunks = [c for c in chunks if c["type"] == "TEXT"]
        assert len(text_chunks) == 1
        assert text_chunks[0]["content"] == "Hello from tuple"

    @pytest.mark.asyncio
    async def test_updates_tuple_ignored(self, processor):
        """'updates' events carry final state — StreamProcessor must skip them."""
        events = [("updates", {"agent_node": {"messages": []}})]
        chunks = await _collect(processor, events)
        text_chunks = [c for c in chunks if c["type"] == "TEXT"]
        assert len(text_chunks) == 0
        assert chunks[-1] == {"type": "STATUS", "content": "COMPLETE"}


# ── Sentence boundary splitting ────────────────────────────────────────────

class TestSentenceBuffering:
    """Unit tests for the sentence boundary splitting logic."""

    def test_splits_on_period(self):
        from app.agent.stream_processor import _SENTENCE_BOUNDARY

        text = "First sentence. Second sentence. Third."
        parts = _SENTENCE_BOUNDARY.split(text)
        assert len(parts) >= 2
        assert "First sentence." in parts[0]

    def test_no_split_mid_sentence(self):
        from app.agent.stream_processor import _SENTENCE_BOUNDARY

        text = "This has no terminal punctuation yet"
        parts = _SENTENCE_BOUNDARY.split(text)
        assert len(parts) == 1
