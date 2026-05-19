"""Summarize tool — extracts first two sentences as a summary."""
import logging
import re

from langchain_core.tools import tool

from app.config.logging_config import get_trace_id
from app.config.metrics import get_metrics

logger = logging.getLogger(__name__)


@tool
async def summarize_text(text: str) -> str:
    """Summarize provided text by extracting the first two sentences."""
    trace = get_trace_id()
    logger.info("[%s] tool=summarize_text length=%d", trace, len(text))
    get_metrics().record_tool_call("summarize_text")
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:2]) if sentences else text[:200]
    return f"Summary: {summary}"
