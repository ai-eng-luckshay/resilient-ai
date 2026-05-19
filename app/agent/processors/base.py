"""
BaseProcessor — abstract contract every processor backend must implement.

The Factory + Strategy pattern here follows the Open/Closed Principle:
new backends (Google ADK, custom A2A orchestrator, etc.) are added by
creating a new subclass and registering it in processor_factory.py —
zero changes to existing code.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.models.schemas import ChatRequest


class BaseProcessor(ABC):
    """
    A processor transforms a ChatRequest into a stream of NDJSON chunks.

    Each chunk is a JSON string terminated by a newline, with the shape:
        {"type": "TEXT|TOOL_CALL|TOOL_RESULT|FAILOVER|STATUS|ERROR",
         "content": "...",
         "metadata": {...}}
    """

    @abstractmethod
    async def stream(self, request: ChatRequest, history: list) -> AsyncIterator[str]:
        """
        Yield NDJSON-encoded stream chunks.

        Args:
            request:  Validated ChatRequest from the REST/A2A layer.
            history:  Existing LangChain message objects for this session.

        Yields:
            NDJSON strings — one per line — for SSE delivery.
        """
        ...
