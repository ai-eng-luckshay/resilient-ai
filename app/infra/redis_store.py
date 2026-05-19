"""
Redis-backed store implementations — production upgrade path.

Swap path, tradeoffs, and key schema documented in docs/architecture.md.

Activation:
  1. Add redis>=5.0 to requirements.in and run pip-compile + pip-sync.
  2. Set USE_REDIS=true and REDIS_URL in .env.
  3. Replace _get_redis_client() below with a real redis.Redis instance.
"""
import json
import time
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Redis client helper
# ---------------------------------------------------------------------------
# In production, replace this with:
#   import redis
#   _client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
# or for async usage:
#   import redis.asyncio as aioredis
#   _client = aioredis.from_url(settings.redis_url, decode_responses=True)

def _get_redis_client():
    raise NotImplementedError(
        "Redis stub — install redis>=5.0, set USE_REDIS=true and REDIS_URL in .env "
        "to activate.  See app/infra/redis_store.py for full wiring instructions."
    )


# ---------------------------------------------------------------------------
# 1. Redis-backed SessionStore
# ---------------------------------------------------------------------------

class RedisSessionStore:
    """
    Replaces: app/services/session_store.py :: SessionStore

    Redis key schema:
        session:{session_id}  →  JSON hash
            {history: [...], system_prompt: "...", created_at: float}
        TTL: SESSION_TTL_SECONDS (default 7200s / 2 h)

    Why this is better in production:
        - History survives uvicorn restarts and rolling deploys.
        - Works across any number of horizontally-scaled workers.
        - Redis native TTL replaces the manual time.time() checks.
        - SCAN-based active_count() works correctly across workers.
    """

    _KEY_PREFIX = "session:"

    def __init__(self, ttl_seconds: int = 7200) -> None:
        self._redis = _get_redis_client()   # swap: inject real client here
        self._ttl = ttl_seconds

    def create(self, system_prompt: str = "") -> str:
        session_id = str(uuid.uuid4())
        payload = json.dumps({
            "history": [],
            "system_prompt": system_prompt,
            "created_at": time.time(),
        })
        # SET session:{id} <json> EX <ttl>
        self._redis.set(f"{self._KEY_PREFIX}{session_id}", payload, ex=self._ttl)
        return session_id

    def get(self, session_id: str) -> dict[str, Any] | None:
        raw = self._redis.get(f"{self._KEY_PREFIX}{session_id}")
        if raw is None:
            return None
        data = json.loads(raw)
        # Refresh TTL on access (sliding expiry)
        self._redis.expire(f"{self._KEY_PREFIX}{session_id}", self._ttl)
        return data

    def update_history(self, session_id: str, history: list) -> None:
        key = f"{self._KEY_PREFIX}{session_id}"
        raw = self._redis.get(key)
        if raw is None:
            return
        data = json.loads(raw)
        data["history"] = history
        self._redis.set(key, json.dumps(data), ex=self._ttl)

    def exists(self, session_id: str) -> bool:
        return self._redis.exists(f"{self._KEY_PREFIX}{session_id}") == 1

    def delete(self, session_id: str) -> None:
        self._redis.delete(f"{self._KEY_PREFIX}{session_id}")

    def active_count(self) -> int:
        # SCAN is O(N) but non-blocking — safe for production
        return sum(1 for _ in self._redis.scan_iter(f"{self._KEY_PREFIX}*"))


# ---------------------------------------------------------------------------
# 2. Redis-backed ContextStore
# ---------------------------------------------------------------------------

class RedisContextStore:
    """
    Replaces: app/a2a/context_store.py :: ContextStore

    Redis key schema:
        a2a:context:{context_id}  →  session_id (plain string)
        TTL: A2A_CONTEXT_TTL_SECONDS (default 86400s / 24 h)

    Why this is better in production:
        - A2A contexts span multiple HTTP calls (multiple task_ids per
          conversation).  If the gateway restarts between tasks, the
          context_id → session_id mapping is lost and the agent forgets
          the conversation.  Redis survives that restart.
        - In a load-balanced deployment, task 1 may hit Worker A and
          task 2 may hit Worker B.  Without shared storage, Worker B
          has no knowledge of the context mapping Worker A created.
    """

    _KEY_PREFIX = "a2a:context:"

    def __init__(self, ttl_seconds: int = 86400) -> None:
        self._redis = _get_redis_client()
        self._ttl = ttl_seconds

    def resolve_session(self, context_id: str, create_fn) -> str:
        key = f"{self._KEY_PREFIX}{context_id}"
        session_id = self._redis.get(key)
        if session_id:
            return session_id
        session_id = create_fn()
        self._redis.set(key, session_id, ex=self._ttl)
        return session_id

    def get_session_id(self, context_id: str) -> str | None:
        return self._redis.get(f"{self._KEY_PREFIX}{context_id}")

    def invalidate(self, context_id: str) -> None:
        self._redis.delete(f"{self._KEY_PREFIX}{context_id}")


# ---------------------------------------------------------------------------
# 3. Redis-backed TaskStore
# ---------------------------------------------------------------------------

class RedisTaskStore:
    """
    Replaces: app/a2a/task_store.py :: InMemoryTaskStore

    Redis key schema:
        a2a:task:{task_id}  →  JSON blob
            {task_id, context_id, session_id, state, artifacts, error, created_at}
        TTL: 3600s (1 h) — tasks are short-lived; no need to persist indefinitely.

    Why this is better in production:
        - A2A clients poll GET /agent/a2a/tasks/{id} after submission.
          If the gateway restarts between submission and polling, the
          in-memory task is gone and the client gets a 404.
        - In a multi-worker setup the worker that created the task may
          not be the one that handles the poll request.
        - Redis APPEND-style pattern lets multiple workers update
          artifact lists atomically via RPUSH on a list key.
    """

    _KEY_PREFIX = "a2a:task:"
    _TTL = 3600  # tasks expire after 1 hour

    def __init__(self) -> None:
        self._redis = _get_redis_client()

    def _key(self, task_id: str) -> str:
        return f"{self._KEY_PREFIX}{task_id}"

    def create(self, context_id: str, session_id: str):
        from app.a2a.task_store import A2ATask  # avoid circular import
        task = A2ATask(
            task_id=str(uuid.uuid4()),
            context_id=context_id,
            session_id=session_id,
        )
        self._redis.set(self._key(task.task_id), json.dumps({
            "task_id": task.task_id,
            "context_id": context_id,
            "session_id": session_id,
            "state": "working",
            "artifacts": [],
            "error": None,
            "created_at": task.created_at,
        }), ex=self._TTL)
        return task

    def _load(self, task_id: str) -> dict | None:
        raw = self._redis.get(self._key(task_id))
        return json.loads(raw) if raw else None

    def _save(self, data: dict) -> None:
        self._redis.set(self._key(data["task_id"]), json.dumps(data), ex=self._TTL)

    def get(self, task_id: str):
        data = self._load(task_id)
        if not data:
            return None
        from app.a2a.task_store import A2ATask
        t = A2ATask(task_id=data["task_id"], context_id=data["context_id"],
                    session_id=data["session_id"])
        t.state = data["state"]
        t.artifacts = data["artifacts"]
        t.error = data["error"]
        t.created_at = data["created_at"]
        return t

    def add_artifact(self, task_id: str, text: str) -> None:
        data = self._load(task_id)
        if data:
            data["artifacts"].append({"type": "text", "text": text,
                                      "index": len(data["artifacts"])})
            self._save(data)

    def complete(self, task_id: str) -> None:
        data = self._load(task_id)
        if data:
            data["state"] = "completed"
            self._save(data)

    def fail(self, task_id: str, reason: str) -> None:
        data = self._load(task_id)
        if data:
            data["state"] = "failed"
            data["error"] = reason
            self._save(data)

    def to_dict(self, task_id: str) -> dict | None:
        data = self._load(task_id)
        if not data:
            return None
        return {
            "id": data["task_id"],
            "contextId": data["context_id"],
            "status": {"state": data["state"], "error": data["error"]},
            "artifacts": data["artifacts"],
        }
