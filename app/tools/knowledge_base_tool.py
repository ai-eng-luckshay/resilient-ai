"""Knowledge base tool — mock KB search over a fixed topic set."""
import logging

from langchain_core.tools import tool

from app.config.logging_config import get_trace_id
from app.config.metrics import get_metrics

logger = logging.getLogger(__name__)

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
    matches = [v for k, v in _MOCK_KNOWLEDGE_BASE.items() if any(w in k for w in q.split())]
    if matches:
        return f"[Knowledge Base] Partial match:\n{matches[0]}"
    return (
        f"No results found for '{query}'. "
        "Try: machine learning, python, microservices, a2a protocol, langgraph."
    )
