"""
ProcessorFactory — resolves a processor backend from a string key.

Registry-based factory keeps the main app code clean: adding a new backend
means registering it here and creating the subclass — nothing else changes.
"""
import logging
from typing import Literal

from app.agent.processors.base_processor import BaseProcessor
from app.agent.processors.langgraph_processor import LangGraphProcessor
from app.agent.processors.google_adk_processor import GoogleADKProcessor

logger = logging.getLogger(__name__)

ProcessorKey = Literal["LANGGRAPH", "GOOGLE_ADK"]

_REGISTRY: dict[str, type[BaseProcessor]] = {
    "LANGGRAPH": LangGraphProcessor,
    "GOOGLE_ADK": GoogleADKProcessor,
}


def get_processor(key: ProcessorKey = "LANGGRAPH") -> BaseProcessor:
    """
    Return an instantiated processor for the given key.

    Args:
        key: One of LANGGRAPH | GOOGLE_ADK

    Returns:
        BaseProcessor instance.

    Raises:
        ValueError if key is not in the registry.
    """
    cls = _REGISTRY.get(key.upper())
    if cls is None:
        raise ValueError(
            f"Unknown processor '{key}'. Available: {list(_REGISTRY.keys())}"
        )
    logger.debug("ProcessorFactory: instantiating %s", cls.__name__)
    return cls()


def available_processors() -> list[str]:
    return list(_REGISTRY.keys())
