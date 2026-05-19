"""
Tool registry — single source of truth for all LLM-bound tools.

Import ALL_TOOLS wherever a complete tool list is needed.
Each tool lives in its own module for single-responsibility clarity.
"""
from app.tools.calculate_tool import calculate
from app.tools.weather_tool import get_weather
from app.tools.knowledge_base_tool import search_knowledge_base
from app.tools.summarize_tool import summarize_text

ALL_TOOLS = [calculate, search_knowledge_base, get_weather, summarize_text]
TOOL_NAMES = {t.name for t in ALL_TOOLS}
