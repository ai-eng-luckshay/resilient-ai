"""Weather tool — mock weather data for demo cities."""
import logging

from langchain_core.tools import tool

from app.config.logging_config import get_trace_id
from app.config.metrics import get_metrics

logger = logging.getLogger(__name__)

_MOCK_WEATHER = {
    "london": {"temp": "14°C", "condition": "Cloudy", "humidity": "75%"},
    "new york": {"temp": "22°C", "condition": "Sunny", "humidity": "55%"},
    "tokyo": {"temp": "18°C", "condition": "Partly cloudy", "humidity": "65%"},
    "paris": {"temp": "16°C", "condition": "Light rain", "humidity": "80%"},
    "sydney": {"temp": "25°C", "condition": "Clear", "humidity": "45%"},
}


@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city (demo data)."""
    trace = get_trace_id()
    logger.info("[%s] tool=get_weather city=%r", trace, city)
    get_metrics().record_tool_call("get_weather")
    data = _MOCK_WEATHER.get(city.lower().strip())
    if data:
        return (
            f"Weather in {city.title()}: {data['condition']}, "
            f"{data['temp']}, Humidity {data['humidity']} (demo data)"
        )
    return (
        f"No weather data for '{city}'. "
        "Available cities: London, New York, Tokyo, Paris, Sydney."
    )
