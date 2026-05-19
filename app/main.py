"""
Resilient AI — FastAPI application entry point.

Startup sequence:
  1. Configure logging (4 categories, hourly rotation, async queue dispatch)
  2. Start QueueListener background threads
  3. Eagerly initialise LLM registry (fails fast if no API keys)
  4. Initialise AgentRunnerManager (builds middleware + builder + compiles graph)
  5. Mount middleware (LIFO order: last added = outermost)
  6. Mount routers

Shutdown sequence:
  1. Flush and stop all QueueListener threads

Middleware stack (outermost → innermost):
  CORSMiddleware → LoggerMiddleware → (request handler)
"""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.a2a.server import router as a2a_router
from app.agent.registry.llm_registry import get_llm_registry
from app.agent.runner.runner_manager import AgentRunnerManager
from app.config.logging_config import (
    configure_logging,
    get_logger,
    start_logging_listener,
    stop_logging_listener,
)
from app.config.settings import get_settings
from app.api.v1.chat_router import router as agent_router
from app.api.v1.health_router import router as health_router
from app.api.middleware.logging_middleware import LoggerMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────
    settings = get_settings()
    configure_logging(log_level=settings.log_level)
    start_logging_listener()

    logger.info(
        "Starting %s v%s [env=%s]", settings.app_name, settings.app_version, settings.env
    )
    logger.info(
        "Log files → logs/YYYY-MM-DD/  "
        "(SYSTEM | ERROR | REQ_RESP | UI) rotated hourly"
    )

    registry = get_llm_registry()
    available = registry.available()
    if not available:
        logger.warning(
            "No LLM providers registered. Set OPENAI_API_KEY or GEMINI_API_KEY in .env"
        )
    else:
        logger.info("LLM providers available: %s", available)

    # Build middleware + agent builder + compile LangGraph workflow
    AgentRunnerManager.initialize()

    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down %s — flushing log queues…", settings.app_name)
    stop_logging_listener()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Resilient AI — LangGraph-based agent with dynamic tool binding, "
            "LLM failover chain, streaming sentence buffering, and A2A protocol support."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware (added in reverse: last-added is outermost)
    app.add_middleware(LoggerMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health_router)
    app.include_router(agent_router)
    app.include_router(a2a_router)

    return app


app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.api_port,
        log_config=None,  # we manage logging ourselves
    )
