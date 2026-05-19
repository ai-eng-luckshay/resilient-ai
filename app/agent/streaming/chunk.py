"""Streaming chunk helpers — NDJSON serialisation and content extraction."""
import json
import re
from typing import Any

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
