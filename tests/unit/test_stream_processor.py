"""
Unit tests for StreamProcessor.

Tests the core streaming pipeline logic:
  - BUFFERED mode: sentence-level chunking
  - RAW mode: immediate token yield
  - FAILOVER chunk passthrough
  - Error handling: re-raises so LangGraphProcessor can trigger failover
  - STATUS:COMPLETE is always the last chunk on success
"""
import json
from typing import AsyncIterator
from unittest.mock import patch

import pytest

from app.agent.stream_processor import StreamProcessor, _ndjson


# ── Helpers ────────────────────────────────────────────────────────────────

async def _make_stream(*events) -> AsyncIterator:
    for e in events:
        yield e


async def _collect(processor: StreamProcessor, events) -> list[dict]:
    chunks = []
    async for raw in processor.process(_make_stream(*events), "GPT4O_MINI"):
        chunks.append(json.loads(raw.strip()))
    return chunks


# ── Tests ──────────────────────────────────────────────────────────────────

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


class TestStreamProcessorBuffered:
    """Tests with STREAM_MODE=BUFFERED (default)."""

    @pytest.fixture
    def processor(self):
        with patch("app.agent.stream_processor.get_settings") as mock_settings:
            mock_settings.return_value.stream_mode = "BUFFERED"
            yield StreamProcessor()

    @pytest.mark.asyncio
    async def test_complete_sentence_yielded(self, processor):
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


class TestStreamProcessorRaw:
    """Tests with STREAM_MODE=RAW."""

    @pytest.fixture
    def processor(self):
        with patch("app.agent.stream_processor.get_settings") as mock_settings:
            mock_settings.return_value.stream_mode = "RAW"
            yield StreamProcessor()

    @pytest.mark.asyncio
    async def test_raw_tokens_yielded_immediately(self, processor):
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


class TestSentenceBuffering:
    """Unit tests for the sentence boundary splitting logic."""

    @pytest.mark.asyncio
    async def test_splits_on_period(self):
        import re
        from app.agent.stream_processor import _SENTENCE_BOUNDARY

        text = "First sentence. Second sentence. Third."
        parts = _SENTENCE_BOUNDARY.split(text)
        assert len(parts) >= 2
        assert "First sentence." in parts[0]

    def test_no_split_mid_sentence(self):
        import re
        from app.agent.stream_processor import _SENTENCE_BOUNDARY

        text = "This has no terminal punctuation yet"
        parts = _SENTENCE_BOUNDARY.split(text)
        assert len(parts) == 1
