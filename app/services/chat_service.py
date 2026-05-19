"""
ChatService: session lifecycle + message history management.

Responsible for:
  - Creating sessions
  - Loading history (LangChain message objects)
  - Appending new messages post-stream
  - Bridging session_store ↔ agent layer
"""
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config.logger import get_trace_id
from app.config.metrics import get_metrics
from app.services.session_store import get_session_store

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self) -> None:
        self._store = get_session_store()

    def create_session(self, system_prompt: str = "") -> str:
        session_id = self._store.create(system_prompt=system_prompt)
        logger.info("[%s] Session created: %s", get_trace_id(), session_id)
        return session_id

    def load_history(self, session_id: str) -> list:
        """
        Returns LangChain message objects for the session.
        Includes SystemMessage prepended if system_prompt is set.
        """
        entry = self._store.get(session_id)
        if entry is None:
            return []
        get_metrics().record_session_hit()

        history = list(entry.get("history", []))
        system_prompt = entry.get("system_prompt", "")

        if not history and system_prompt:
            history = [SystemMessage(content=system_prompt)]

        logger.debug(
            "[%s] Loaded %d messages for session %s",
            get_trace_id(), len(history), session_id,
        )
        return history

    def save_turn(self, session_id: str, user_message: str, assistant_reply: str) -> None:
        entry = self._store.get(session_id)
        if entry is None:
            logger.warning("[%s] Cannot save turn — session %s not found", get_trace_id(), session_id)
            return
        history = list(entry.get("history", []))
        if not history and entry.get("system_prompt"):
            history = [SystemMessage(content=entry["system_prompt"])]
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=assistant_reply))
        self._store.update_history(session_id, history)
        logger.debug(
            "[%s] Saved turn for session %s | history_len=%d",
            get_trace_id(), session_id, len(history),
        )

    def session_exists(self, session_id: str) -> bool:
        exists = self._store.exists(session_id)
        if not exists:
            get_metrics().record_session_miss()
        return exists

    def get_display_history(self, session_id: str) -> list[dict[str, str]]:
        """Return chat history as plain dicts for UI consumption.

        Skips SystemMessage — only user/assistant turns are included.
        Each item: {"role": "user"|"assistant", "content": "..."}
        """
        entry = self._store.get(session_id)
        if entry is None:
            return []
        result: list[dict[str, str]] = []
        for msg in entry.get("history", []):
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": str(msg.content)})
            # SystemMessage intentionally skipped
        return result

    def list_sessions(self) -> list[dict]:
        return self._store.list_all()

    def delete_session(self, session_id: str) -> None:
        self._store.delete(session_id)
        logger.info("[%s] Session deleted: %s", get_trace_id(), session_id)
