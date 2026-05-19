"""Abstract interface for session storage (Repository Pattern)."""
from abc import ABC, abstractmethod
from typing import Any


class ISessionRepository(ABC):
    """
    Interface contract for all session store implementations.
    Concrete classes: InMemorySessionRepository, RedisSessionRepository.
    """

    @abstractmethod
    def create(self, system_prompt: str = "") -> str: ...

    @abstractmethod
    def get(self, session_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def update_history(self, session_id: str, history: list) -> None: ...

    @abstractmethod
    def exists(self, session_id: str) -> bool: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...

    @abstractmethod
    def list_all(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def active_count(self) -> int: ...
