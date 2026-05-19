import ast
import logging
import math
import operator
import re

from langchain_core.tools import tool

from app.config.logger import get_trace_id
from app.config.metrics import get_metrics

logger = logging.getLogger(__name__)

# Safe operators for the calculator tool
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

_MOCK_KNOWLEDGE_BASE = {
    "machine learning": (
        "Machine learning is a branch of AI where systems learn from data. "
        "Key paradigms: supervised learning (labeled data), unsupervised learning (unlabeled), "
        "and reinforcement learning (reward signals). Common algorithms include linear regression, "
        "decision trees, neural networks, and SVMs."
    ),
    "python": (
        "Python is a high-level, interpreted programming language known for readability. "
        "It supports multiple paradigms: procedural, OOP, and functional. "
        "Widely used in web development (FastAPI, Django), data science (pandas, numpy), "
        "and AI/ML (PyTorch, TensorFlow)."
    ),
    "microservices": (
        "Microservices architecture decomposes applications into small, independent services "
        "that communicate via APIs. Benefits: independent deployment, technology diversity, fault isolation. "
        "Challenges: distributed systems complexity, network latency, data consistency."
    ),
    "a2a protocol": (
        "Agent-to-Agent (A2A) protocol is an open standard for AI agents to communicate, "
        "delegate tasks, and exchange artifacts. It uses JSON-RPC 2.0 with task/artifact lifecycles. "
        "Enables multi-agent orchestration without vendor lock-in."
    ),
    "langgraph": (
        "LangGraph is a framework for building stateful, multi-actor AI applications using graphs. "
        "Agents are modelled as nodes; edges define control flow including conditional routing and loops. "
        "State is a TypedDict shared across all nodes, enabling complex multi-step reasoning."
    ),
}

_MOCK_WEATHER = {
    "london": {"temp": "14°C", "condition": "Cloudy", "humidity": "75%"},
    "new york": {"temp": "22°C", "condition": "Sunny", "humidity": "55%"},
    "tokyo": {"temp": "18°C", "condition": "Partly cloudy", "humidity": "65%"},
    "paris": {"temp": "16°C", "condition": "Light rain", "humidity": "80%"},
    "sydney": {"temp": "25°C", "condition": "Clear", "humidity": "45%"},
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        allowed_funcs = {"sqrt": math.sqrt, "abs": abs, "round": round}
        if func_name in allowed_funcs:
            args = [_safe_eval(a) for a in node.args]
            return allowed_funcs[func_name](*args)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, %, sqrt(), abs(), round()."""
    trace = get_trace_id()
    logger.info("[%s] tool=calculate expr=%r", trace, expression)
    get_metrics().record_tool_call("calculate")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@tool
async def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information on a topic."""
    trace = get_trace_id()
    logger.info("[%s] tool=search_knowledge_base query=%r", trace, query)
    get_metrics().record_tool_call("search_knowledge_base")

    q = query.lower().strip()
    for key, content in _MOCK_KNOWLEDGE_BASE.items():
        if key in q or q in key:
            return f"[Knowledge Base] {key.title()}:\n{content}"

    # Fuzzy fallback: return any partial keyword match
    matches = [v for k, v in _MOCK_KNOWLEDGE_BASE.items() if any(w in k for w in q.split())]
    if matches:
        return f"[Knowledge Base] Partial match:\n{matches[0]}"
    return (
        f"No results found for '{query}'. "
        "Try: machine learning, python, microservices, a2a protocol, langgraph."
    )


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


@tool
async def summarize_text(text: str) -> str:
    """Summarize provided text by extracting the first two sentences."""
    trace = get_trace_id()
    logger.info("[%s] tool=summarize_text length=%d", trace, len(text))
    get_metrics().record_tool_call("summarize_text")

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    summary = " ".join(sentences[:2]) if sentences else text[:200]
    return f"Summary: {summary}"


ALL_TOOLS = [calculate, search_knowledge_base, get_weather, summarize_text]
TOOL_NAMES = {t.name for t in ALL_TOOLS}
