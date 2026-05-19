"""Abstract interface for A2A context storage (Repository Pattern)."""
from abc import ABC, abstractmethod
from typing import Callable


class IContextRepository(ABC):
    """
    Interface contract for all A2A context store implementations.
    Concrete classes: InMemoryContextRepository, RedisContextRepository.
    """

    @abstractmethod
    def resolve_session(self, context_id: str, create_fn: Callable[[], str]) -> str: ...

    @abstractmethod
    def get_session_id(self, context_id: str) -> str | None: ...

    @abstractmethod
    def invalidate(self, context_id: str) -> None: ...
