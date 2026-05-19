"""
In-memory session store with TTL.

Set USE_REDIS=true in .env to swap to the Redis-backed implementation.
See docs/architecture.md for tradeoffs and key schema.
"""
import time
import uuid
from functools import lru_cache
from typing import Any

from app.repositories.base.session_repository import ISessionRepository


class SessionStore(ISessionRepository):
    def __init__(self, ttl_seconds: int = 7200) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._ttl = ttl_seconds

    def create(self, system_prompt: str = "") -> str:
        session_id = str(uuid.uuid4())
        self._store[session_id] = {
            "history": [],
            "system_prompt": system_prompt,
            "created_at": time.time(),
            "last_accessed": time.time(),
        }
        return session_id

    def get(self, session_id: str) -> dict[str, Any] | None:
        entry = self._store.get(session_id)
        if entry is None:
            return None
        if time.time() - entry["created_at"] > self._ttl:
            del self._store[session_id]
            return None
        entry["last_accessed"] = time.time()
        return entry

    def update_history(self, session_id: str, history: list) -> None:
        if session_id in self._store:
            self._store[session_id]["history"] = history
            self._store[session_id]["last_accessed"] = time.time()

    def exists(self, session_id: str) -> bool:
        return self.get(session_id) is not None

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def list_all(self) -> list[dict[str, Any]]:
        """Return summary metadata for all non-expired sessions, sorted newest-first."""
        now = time.time()
        result = []
        for sid, entry in list(self._store.items()):
            if now - entry["created_at"] <= self._ttl:
                result.append({
                    "session_id": sid,
                    "message_count": len(entry.get("history", [])),
                    "system_prompt": entry.get("system_prompt", ""),
                    "created_at": entry["created_at"],
                    "last_accessed": entry["last_accessed"],
                })
        result.sort(key=lambda x: x["last_accessed"], reverse=True)
        return result

    def active_count(self) -> int:
        now = time.time()
        return sum(
            1 for v in self._store.values()
            if now - v["created_at"] <= self._ttl
        )


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    from app.config.settings import get_settings
    settings = get_settings()
    if settings.use_redis:
        # Swap to Redis-backed store for multi-worker / persistent deployments.
        # See app/repositories/redis/session_repository.py — RedisSessionStore implements
        # the same interface; activate by setting USE_REDIS=true and REDIS_URL in .env.
        from app.repositories.redis.session_repository import RedisSessionStore
        return RedisSessionStore(ttl_seconds=settings.session_ttl_seconds)  # type: ignore[return-value]
    return SessionStore(ttl_seconds=settings.session_ttl_seconds)
