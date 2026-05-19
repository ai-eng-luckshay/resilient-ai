"""
Unit tests for LLMRegistry.

Tests:
  - Provider registration with valid keys
  - get() raises KeyError for unknown providers
  - get_with_failover() returns first available provider
  - get_with_failover() raises RuntimeError when all exhausted
  - available() reflects registered providers
"""
from unittest.mock import MagicMock

import pytest

from app.agent.registry.llm_registry import LLMRegistry


class TestLLMRegistryGet:
    @pytest.fixture
    def registry_with_openai(self):
        mock_llm = MagicMock()
        registry = LLMRegistry.__new__(LLMRegistry)
        registry._providers = {"GPT4O_MINI": mock_llm, "GPT35_TURBO": mock_llm}
        return registry, mock_llm

    def test_get_known_provider(self, registry_with_openai):
        registry, mock_llm = registry_with_openai
        result = registry.get("GPT4O_MINI")
        assert result is mock_llm

    def test_get_unknown_raises_key_error(self, registry_with_openai):
        registry, _ = registry_with_openai
        with pytest.raises(KeyError, match="UNKNOWN"):
            registry.get("UNKNOWN")

    def test_available_returns_list(self, registry_with_openai):
        registry, _ = registry_with_openai
        providers = registry.available()
        assert "GPT4O_MINI" in providers
        assert "GPT35_TURBO" in providers


class TestLLMFailoverChain:
    @pytest.fixture
    def registry_partial(self):
        """Only GPT35_TURBO registered (GPT4O_MINI missing)."""
        mock_llm = MagicMock()
        registry = LLMRegistry.__new__(LLMRegistry)
        registry._providers = {"GPT35_TURBO": mock_llm}
        return registry, mock_llm

    def test_failover_skips_missing_provider(self, registry_partial):
        registry, mock_llm = registry_partial
        llm, key = registry.get_with_failover(["GPT4O_MINI", "GPT35_TURBO"])
        assert key == "GPT35_TURBO"
        assert llm is mock_llm

    def test_failover_exhausted_raises_runtime_error(self):
        registry = LLMRegistry.__new__(LLMRegistry)
        registry._providers = {}
        with pytest.raises(RuntimeError, match="No registered providers in chain"):
            registry.get_with_failover(["GPT4O_MINI", "GPT35_TURBO"])

    def test_preferred_provider_returned_first(self):
        mock_gpt4 = MagicMock()
        mock_gpt35 = MagicMock()
        registry = LLMRegistry.__new__(LLMRegistry)
        registry._providers = {"GPT4O_MINI": mock_gpt4, "GPT35_TURBO": mock_gpt35}
        llm, key = registry.get_with_failover(["GPT4O_MINI", "GPT35_TURBO"])
        assert key == "GPT4O_MINI"
        assert llm is mock_gpt4
