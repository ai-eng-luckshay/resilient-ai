import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config.logging_config import (
    get_logger,
    get_req_resp_logger,
    set_trace_id,
    set_request_info,
)
from app.config.metrics import get_metrics

# System logger for lifecycle events (startup messages, errors)
sys_logger = get_logger(__name__)
# Dedicated req/resp logger → gateway_REQ_RESP_{date}_{hour}.log
rr_logger = get_req_resp_logger()

# Paths excluded from request/response logging (polled every 2 s by the UI)
_SILENT_PATHS = {"/health", "/metrics"}


class LoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in _SILENT_PATHS:
            return await call_next(request)

        trace_id = request.headers.get("x-trace-id", str(uuid.uuid4())[:8])
        client_ip = request.client.host if request.client else "-"

        set_trace_id(trace_id)
        set_request_info(request.method, request.url.path, client_ip)
        request.state.trace_id = trace_id

        start = time.perf_counter()

        # Log incoming request
        rr_logger.info(
            "REQUEST  method=%s path=%s client=%s",
            request.method,
            request.url.path,
            client_ip,
        )

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start) * 1000

        # Log outgoing response
        rr_logger.info(
            "RESPONSE method=%s path=%s status=%d latency=%.1fms",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )

        # Warn on slow requests (>2 s) — captured by ERROR log via root handler
        if latency_ms > 2000:
            sys_logger.warning(
                "SLOW_REQUEST method=%s path=%s latency=%.1fms",
                request.method,
                request.url.path,
                latency_ms,
            )

        # Record metrics for chat / stream endpoints only
        if any(p in request.url.path for p in ("/chat", "/stream")):
            metrics = get_metrics()
            metrics.record_request(latency_ms)

        response.headers["x-trace-id"] = trace_id
        return response
