from typing import Any, Literal
from pydantic import BaseModel, Field
import uuid


class SessionInitRequest(BaseModel):
    system_prompt: str = "You are a helpful AI assistant."


class SessionInitResponse(BaseModel):
    session_id: str
    message: str = "Session initialised."


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "GPT4O_MINI"
    processor: Literal["LANGGRAPH", "GOOGLE_ADK"] = "LANGGRAPH"
    system_prompt: str = "You are a helpful AI assistant."


class StreamChunk(BaseModel):
    type: Literal["TEXT", "TOOL_CALL", "TOOL_RESULT", "FAILOVER", "STATUS", "ERROR"]
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionListItem(BaseModel):
    session_id: str
    message_count: int
    system_prompt: str
    created_at: float
    last_accessed: float


class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]
    total: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    env: str


# Internal context object threaded through the agent workflow
class RequestContext(BaseModel):
    session_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    system_prompt: str = "You are a helpful AI assistant."
    allowed_tools: list[str] = Field(default_factory=list)
