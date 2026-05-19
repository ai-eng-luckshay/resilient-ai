"""
REST API endpoints:
  POST /v1/session/init   — create a new session
  POST /v1/chat           — non-streaming chat response
  POST /v1/stream         — SSE streaming chat response
"""
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agent.processor_factory import get_processor
from app.config.logger import get_trace_id
from app.config.metrics import get_metrics
from app.models.schemas import (
    ChatRequest,
    SessionInitRequest,
    SessionInitResponse,
    SessionListResponse,
)
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["agent"])
_chat_service = ChatService()


@router.post("/session/init", response_model=SessionInitResponse)
async def init_session(body: SessionInitRequest, request: Request) -> SessionInitResponse:
    session_id = _chat_service.create_session(system_prompt=body.system_prompt)
    logger.info("[%s] Session init: %s", get_trace_id(), session_id)
    return SessionInitResponse(session_id=session_id)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    sessions = _chat_service.list_sessions()
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str) -> dict:
    if not _chat_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    history = _chat_service.get_display_history(session_id)
    return {"session_id": session_id, "history": history}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict:
    if not _chat_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    _chat_service.delete_session(session_id)
    logger.info("[%s] Session deleted via API: %s", get_trace_id(), session_id)
    return {"deleted": session_id}


@router.post("/stream")
async def stream_chat(body: ChatRequest, request: Request) -> StreamingResponse:
    """SSE streaming endpoint — yields NDJSON chunks as they arrive."""
    if not _chat_service.session_exists(body.session_id):
        raise HTTPException(status_code=404, detail=f"Session '{body.session_id}' not found or expired.")

    history = _chat_service.load_history(body.session_id)
    processor = get_processor(body.processor)

    accumulated: list[str] = []

    async def event_generator() -> AsyncIterator[bytes]:
        try:
            async for chunk in processor.stream(body, history):
                if await request.is_disconnected():
                    logger.info("[%s] Client disconnected — stream cancelled", get_trace_id())
                    return
                accumulated.append(chunk)
                yield f"data: {chunk}\n\n".encode()
        except NotImplementedError:
            # Stub processor — already yielded info chunks, just close stream
            pass
        except Exception as exc:
            logger.error("[%s] Stream error: %s", get_trace_id(), exc)
            get_metrics().record_error()
            error_chunk = json.dumps({"type": "ERROR", "content": str(exc)}) + "\n"
            yield f"data: {error_chunk}\n\n".encode()
        finally:
            # Persist the assistant's reply to session history
            try:
                full_reply = _extract_text_content(accumulated)
                if full_reply:
                    _chat_service.save_turn(body.session_id, body.message, full_reply)
            except Exception as save_exc:
                logger.warning("[%s] Failed to save turn: %s", get_trace_id(), save_exc)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "x-trace-id": get_trace_id(),
        },
    )


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> dict:
    """Non-streaming endpoint — collects full response then returns."""
    if not _chat_service.session_exists(body.session_id):
        raise HTTPException(status_code=404, detail=f"Session '{body.session_id}' not found or expired.")

    history = _chat_service.load_history(body.session_id)
    processor = get_processor(body.processor)

    chunks = []
    try:
        async for chunk in processor.stream(body, history):
            chunks.append(chunk)
    except NotImplementedError:
        pass

    full_reply = _extract_text_content(chunks)
    _chat_service.save_turn(body.session_id, body.message, full_reply)

    return {
        "session_id": body.session_id,
        "reply": full_reply,
        "trace_id": get_trace_id(),
        "provider": body.model,
    }


def _extract_text_content(chunks: list[str]) -> str:
    parts = []
    for raw in chunks:
        try:
            parsed = json.loads(raw.strip())
            if parsed.get("type") == "TEXT":
                parts.append(parsed.get("content", ""))
        except Exception:
            pass
    return " ".join(parts).strip()
