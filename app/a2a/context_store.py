"""
ContextStore: binds A2A context_id → session_id.

Maintains conversation continuity across A2A tasks that share the same
context_id. Set USE_REDIS=true in .env to swap to Redis-backed storage.
See docs/architecture.md for tradeoffs.
"""
import time
import uuid
from functools import lru_cache


class ContextStore:
    def __init__(self, ttl_seconds: int = 86400) -> None:
        self._store: dict[str, dict] = {}
        self._ttl = ttl_seconds

    def resolve_session(self, context_id: str, create_fn) -> str:
        """
        Return existing session_id for context_id, or call create_fn() to
        create a new one and persist the mapping.

        Args:
            context_id:  A2A context identifier (from JSON-RPC params).
            create_fn:   Zero-arg callable that returns a new session_id.
        """
        entry = self._store.get(context_id)
        if entry and time.time() - entry["created_at"] < self._ttl:
            return entry["session_id"]

        session_id = create_fn()
        self._store[context_id] = {
            "session_id": session_id,
            "created_at": time.time(),
        }
        return session_id

    def get_session_id(self, context_id: str) -> str | None:
        entry = self._store.get(context_id)
        if entry and time.time() - entry["created_at"] < self._ttl:
            return entry["session_id"]
        return None

    def invalidate(self, context_id: str) -> None:
        self._store.pop(context_id, None)


@lru_cache(maxsize=1)
def get_context_store() -> ContextStore:
    from app.config.settings import get_settings
    settings = get_settings()
    if settings.use_redis:
        # Swap to Redis-backed store — critical for multi-worker deployments
        # where A2A task 1 and task 2 (same context_id) may hit different workers.
        # See app/infra/redis_store.py — RedisContextStore implements the same interface.
        from app.infra.redis_store import RedisContextStore
        return RedisContextStore(ttl_seconds=settings.a2a_context_ttl_seconds)  # type: ignore[return-value]
    return ContextStore(ttl_seconds=settings.a2a_context_ttl_seconds)
