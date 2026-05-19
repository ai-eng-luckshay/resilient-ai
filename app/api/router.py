"""
Central API router — aggregates all sub-routers for clean main.py wiring.
"""
from app.api.v1.chat_router import router as v1_chat_router
from app.api.v1.health_router import router as health_router
from app.a2a.server import router as a2a_router

__all__ = ["v1_chat_router", "health_router", "a2a_router"]
