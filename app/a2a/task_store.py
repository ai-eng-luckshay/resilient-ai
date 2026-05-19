"""
In-memory A2A task store.

Stores task state (WORKING → COMPLETE | FAILED) and artifact lists.
The a2a-sdk defines the Task/Artifact schema; this module implements
the persistence layer the SDK delegates to.
"""
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal


TaskState = Literal["working", "completed", "failed", "canceled"]


@dataclass
class A2ATask:
    task_id: str
    context_id: str
    session_id: str
    state: TaskState = "working"
    artifacts: list[dict] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class InMemoryTaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, A2ATask] = {}

    def create(self, context_id: str, session_id: str) -> A2ATask:
        task = A2ATask(
            task_id=str(uuid.uuid4()),
            context_id=context_id,
            session_id=session_id,
        )
        self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> A2ATask | None:
        return self._tasks.get(task_id)

    def add_artifact(self, task_id: str, text: str) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.artifacts.append({"type": "text", "text": text, "index": len(task.artifacts)})

    def complete(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.state = "completed"

    def fail(self, task_id: str, reason: str) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.state = "failed"
            task.error = reason

    def to_dict(self, task_id: str) -> dict | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        return {
            "id": task.task_id,
            "contextId": task.context_id,
            "status": {"state": task.state, "error": task.error},
            "artifacts": task.artifacts,
        }


@lru_cache(maxsize=1)
def get_task_store() -> InMemoryTaskStore:
    from app.config.settings import get_settings
    if get_settings().use_redis:
        # Swap to Redis-backed store — A2A clients poll for task results after
        # submission; a process restart between submit and poll returns 404 with
        # the in-memory store.  Redis survives restarts and works across workers.
        # See app/infra/redis_store.py — RedisTaskStore implements the same interface.
        from app.repositories.redis.task_repository import RedisTaskStore
        return RedisTaskStore()  # type: ignore[return-value]
    return InMemoryTaskStore()
