"""
BaseAgentBuilder — abstract base class for all agent builders.

Mirrors the parent project's BaseAgentBuilder pattern:
  - Enforces a single contract: subclasses implement _create_workflow()
  - Calls _create_workflow() in __init__ so the compiled graph is ready
    immediately after construction
  - Provides get_workflow() and draw_graph() as shared utilities

Adding a new agent backend means subclassing this, implementing
_create_workflow(), and registering the builder — nothing else changes.
"""
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseAgentBuilder(ABC):
    """
    Abstract base for LangGraph workflow builders.

    Subclass contract:
        _create_workflow()  →  return a compiled LangGraph graph

    Shared behaviour (inherited, not overridden):
        get_workflow()   →  returns the compiled graph
        draw_graph()     →  logs an ASCII representation at startup
    """

    def __init__(self) -> None:
        self.workflow = self._create_workflow()

    @abstractmethod
    def _create_workflow(self):
        """Build, compile, and return the LangGraph StateGraph."""
        ...

    def get_workflow(self):
        return self.workflow

    def draw_graph(self) -> None:
        """Log an ASCII graph of the compiled workflow — useful at startup."""
        try:
            ascii_graph = self.workflow.get_graph(xray=1).draw_ascii()
            logger.info("Workflow graph:\n%s", ascii_graph)
        except Exception as exc:
            logger.warning("Could not render workflow graph: %s", exc)
