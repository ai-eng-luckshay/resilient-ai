"""
StreamProcessor: converts raw LangGraph event stream into NDJSON chunks.

Chunk types: TEXT | TOOL_RESULT | FAILOVER | STATUS | ERROR
Streaming modes: BUFFERED (sentence-level) | RAW (every token)
See docs/architecture.md for mode tradeoffs.
"""
import json
import logging
import re
from typing import AsyncIterator, Any

from langchain_core.messages import AIMessageChunk, ToolMessage

from app.config.logger import get_trace_id
from app.config.metrics import get_metrics
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Sentence boundary pattern — splits after terminal punctuation followed by whitespace
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;\n—,])\s+")


def _extract_text(content) -> str:
    """
    Extract plain text from AIMessageChunk.content.

    LangChain 1.x (with langchain-google-genai 3.x) returns content as a
    list of typed content blocks for thinking / multimodal models:
        [{'type': 'text', 'text': 'Hello!', 'index': 0}, ...]
    Earlier versions (and non-thinking models) return a plain str.
    This helper normalises both into a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def _ndjson(chunk_type: str, content: str = "", **metadata) -> str:
    payload: dict[str, Any] = {"type": chunk_type, "content": content}
    if metadata:
        payload["metadata"] = metadata
    return json.dumps(payload, ensure_ascii=False) + "\n"


class StreamProcessor:
    """
    Translates the (event_type, data) tuples from AgentRunner.stream() into
    a sequence of NDJSON-encoded strings ready for SSE delivery.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def process(
        self,
        event_stream: AsyncIterator[tuple[str, Any]],
        active_provider: str,
    ) -> AsyncIterator[str]:
        sentence_buffer: list[str] = [""]
        metrics = get_metrics()
        trace = get_trace_id()

        try:
            async for event in event_stream:
                # LangGraph 1.0.x: astream with stream_mode=["messages","updates"]
                # yields (mode, data) tuples:
                #   ("messages", (AIMessageChunk, metadata_dict))
                #   ("updates",  {node_name: {state_key: value}})
                if isinstance(event, tuple) and len(event) == 2:
                    mode, data = event
                    if mode == "messages":
                        # data is (chunk, metadata) — unwrap the chunk
                        chunk_msg = data[0] if isinstance(data, tuple) else data
                        msgs = chunk_msg if isinstance(chunk_msg, list) else [chunk_msg]
                        async for chunk in self._handle_messages_event(
                            msgs, sentence_buffer, trace
                        ):
                            yield chunk
                    # "updates" mode — skip, full content already in messages stream
                elif isinstance(event, dict):
                    # Fallback: older single stream_mode="messages" yields plain dicts
                    msgs = event.get("messages", [])
                    if msgs:
                        async for chunk in self._handle_messages_event(
                            msgs if isinstance(msgs, list) else [msgs],
                            sentence_buffer,
                            trace,
                        ):
                            yield chunk

            # Flush any remaining buffered text
            remaining = sentence_buffer[0].strip()
            if remaining:
                yield _ndjson("TEXT", remaining)

            yield _ndjson("STATUS", "COMPLETE")
            logger.info("[%s] Stream complete", trace)

        except Exception as exc:
            logger.error("[%s] Stream error (will propagate for failover): %s", trace, exc)
            raise

    async def _handle_messages_event(
        self,
        messages: list,
        sentence_buffer: list[str],
        trace: str,
    ) -> AsyncIterator[str]:
        for msg in messages:
            if isinstance(msg, AIMessageChunk) and msg.content:
                content = _extract_text(msg.content)
                if self._settings.stream_mode == "RAW":
                    if content:
                        yield _ndjson("TEXT", content)
                else:
                    async for chunk in self._buffer_sentence(content, sentence_buffer):
                        yield chunk

            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown_tool")
                result_preview = str(msg.content)[:300]
                logger.debug("[%s] Tool result | tool=%s", trace, tool_name)
                yield _ndjson("TOOL_RESULT", result_preview, tool=tool_name)

    async def _handle_updates_event(
        self,
        updates: dict,
        trace: str,
    ) -> AsyncIterator[str]:
        # Updates node gives us final AIMessage objects — skip if already streamed
        # via messages events to avoid duplicate output
        for _node, _state in (updates.items() if isinstance(updates, dict) else []):
            pass  # Intentionally a no-op; full content already yielded via messages
        return
        yield  # make this an async generator

    async def _buffer_sentence(
        self,
        token: str,
        buffer: list[str],
    ) -> AsyncIterator[str]:
        """
        Accumulate tokens in buffer[0]; yield complete sentences when a
        sentence boundary is detected.

        Design note: buffer is a single-element list so it can be mutated
        from the enclosing generator scope without nonlocal.
        """
        buffer[0] += token
        parts = _SENTENCE_BOUNDARY.split(buffer[0])
        # Keep the last (possibly incomplete) part in the buffer
        for sentence in parts[:-1]:
            if sentence.strip():
                yield _ndjson("TEXT", sentence.strip())
        buffer[0] = parts[-1]
