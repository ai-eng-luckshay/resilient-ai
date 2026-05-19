"""
Unit tests for SessionStore.

Tests:
  - create() returns unique session IDs
  - get() returns session after creation
  - get() returns None after TTL expiry
  - update_history() persists message list
  - exists() reflects live session
  - delete() removes session
  - active_count() counts only live sessions
"""
import time
from unittest.mock import patch

import pytest

from app.services.session_store import SessionStore


class TestSessionCreate:
    def test_creates_unique_ids(self):
        store = SessionStore()
        id1 = store.create()
        id2 = store.create()
        assert id1 != id2

    def test_session_retrievable_after_create(self):
        store = SessionStore()
        sid = store.create(system_prompt="Test prompt")
        entry = store.get(sid)
        assert entry is not None
        assert entry["system_prompt"] == "Test prompt"
        assert entry["history"] == []

    def test_exists_true_after_create(self):
        store = SessionStore()
        sid = store.create()
        assert store.exists(sid) is True

    def test_exists_false_for_unknown(self):
        store = SessionStore()
        assert store.exists("nonexistent-id") is False


class TestSessionExpiry:
    def test_get_returns_none_after_ttl(self):
        store = SessionStore(ttl_seconds=1)
        sid = store.create()
        # Fast-forward time
        with patch("app.services.session_store.time") as mock_time:
            mock_time.time.return_value = time.time() + 10
            result = store.get(sid)
        assert result is None

    def test_active_count_excludes_expired(self):
        store = SessionStore(ttl_seconds=1)
        store.create()
        store.create()
        with patch("app.services.session_store.time") as mock_time:
            mock_time.time.return_value = time.time() + 10
            count = store.active_count()
        assert count == 0


class TestSessionHistory:
    def test_update_history_persists(self):
        from langchain_core.messages import HumanMessage, AIMessage
        store = SessionStore()
        sid = store.create()
        messages = [HumanMessage(content="Hi"), AIMessage(content="Hello!")]
        store.update_history(sid, messages)
        entry = store.get(sid)
        assert len(entry["history"]) == 2

    def test_delete_removes_session(self):
        store = SessionStore()
        sid = store.create()
        store.delete(sid)
        assert store.exists(sid) is False

    def test_active_count_accurate(self):
        store = SessionStore(ttl_seconds=3600)
        store.create()
        store.create()
        store.create()
        assert store.active_count() == 3
