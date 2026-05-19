from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Resilient AI"
    app_version: str = "1.0.0"
    env: Literal["dev", "uat", "prod"] = "dev"
    log_level: str = "INFO"

    # LLM API keys — used by the dynamic registry to instantiate providers
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Processor backend: LANGGRAPH | GOOGLE_ADK
    agent_processor: Literal["LANGGRAPH", "GOOGLE_ADK"] = "LANGGRAPH"

    # Preferred LLM provider key — moved to the front of the failover chain.
    # Full chain is built dynamically from llm_provider_map in agent_util.py.
    selected_llm_provider: str = "GPT4O_MINI"

    # Streaming
    stream_mode: Literal["BUFFERED", "RAW"] = "BUFFERED"

    # Session store TTL (seconds)
    session_ttl_seconds: int = 7200

    # A2A context binding TTL (seconds)
    a2a_context_ttl_seconds: int = 86400

    # A2A remote agent URL (used by A2A processor stub)
    remote_agent_url: str = ""

    # Redis — swap all in-memory stores to Redis-backed equivalents.
    # See app/infra/redis_store.py for implementation and tradeoff docs.
    use_redis: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # Server ports
    api_port: int = 8000
    ui_port: int = 8501


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
