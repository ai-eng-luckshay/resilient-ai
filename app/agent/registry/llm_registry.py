"""
LLMRegistry — builds providers from llm_provider_map in agent_util.py.

The master list of models lives in agent_util.llm_provider_map (code).
The env only controls which subset is active via LLM_PROVIDER_CHAIN.
See docs/architecture.md for design rationale.
"""
import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.agent.config.provider_catalog import llm_provider_map
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_PREFIX_OPENAI = "O"
_PREFIX_GOOGLE = "GG"


def _build_providers() -> dict[str, BaseChatModel]:
    settings = get_settings()
    providers: dict[str, BaseChatModel] = {}

    for provider_key, provider_value in llm_provider_map.items():
        # provider_value format: PREFIX_model-name  e.g. "O_gpt-4o-mini"
        prefix, _, model_name = provider_value.partition("_")

        if not model_name:
            logger.warning("Skipping malformed entry '%s': %r", provider_key, provider_value)
            continue

        if prefix == _PREFIX_OPENAI:
            if not settings.openai_api_key:
                logger.debug("Skipping %s — OPENAI_API_KEY not set.", provider_key)
                continue
            try:
                from langchain_openai import ChatOpenAI
                providers[provider_key] = ChatOpenAI(
                    model=model_name,
                    api_key=settings.openai_api_key,
                    temperature=0.2,
                )
                logger.info("Registered OpenAI provider: %s → %s", provider_key, model_name)
            except Exception as exc:
                logger.warning("Failed to register %s (%s): %s", provider_key, model_name, exc)

        elif prefix == _PREFIX_GOOGLE:
            if not settings.gemini_api_key:
                logger.debug("Skipping %s — GEMINI_API_KEY not set.", provider_key)
                continue
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                providers[provider_key] = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=settings.gemini_api_key,
                    temperature=0.2,
                )
                logger.info(
                    "Registered Google provider: %s → %s | key=...%s",
                    provider_key, model_name, settings.gemini_api_key[-6:],
                )
            except Exception as exc:
                logger.warning("Failed to register %s (%s): %s", provider_key, model_name, exc)

        else:
            logger.warning("Unknown prefix '%s' for provider '%s' — skipping.", prefix, provider_key)

    if not providers:
        logger.error(
            "No LLM providers registered. "
            "Check OPENAI_API_KEY / GEMINI_API_KEY and llm_provider_map in agent_util.py."
        )
    return providers


class LLMRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, BaseChatModel] = _build_providers()

    def get(self, key: str) -> BaseChatModel:
        if key not in self._providers:
            raise KeyError(
                f"LLM provider '{key}' is not registered. "
                f"Available: {list(self._providers.keys())}"
            )
        return self._providers[key]

    def available(self) -> list[str]:
        return list(self._providers.keys())

    def get_with_failover(self, chain: list[str]) -> tuple[BaseChatModel, str]:
        for key in chain:
            if key not in self._providers:
                logger.debug("Provider '%s' not in registry, skipping.", key)
                continue
            return self._providers[key], key
        raise RuntimeError(
            f"No registered providers in chain: {chain}. "
            f"Registered: {list(self._providers.keys())}"
        )


@lru_cache(maxsize=1)
def get_llm_registry() -> LLMRegistry:
    return LLMRegistry()
