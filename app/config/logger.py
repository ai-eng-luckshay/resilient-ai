"""
Logging configuration for Resilient AI.

Log structure:
    logs/
      YYYY-MM-DD/
        gateway_SYSTEM_{date}_{hour}.log     — app lifecycle, startup, LLM events
        gateway_ERROR_{date}_{hour}.log      — WARNING and above from all loggers
        gateway_REQ_RESP_{date}_{hour}.log   — every HTTP request/response (middleware)
        gateway_UI_{date}_{hour}.log         — Streamlit / UI layer events

Rotation:   Hourly  (new file every hour, e.g. _2026-05-19_14.log)
Pattern:    Async   (QueueHandler → QueueListener background thread per category)
            Non-blocking writes; log calls never slow down request handling.

Usage:
    from app.config.logger import configure_logging, get_logger, get_req_resp_logger
    from app.config.logger import start_logging_listener, stop_logging_listener
    from app.config.logger import set_trace_id, get_trace_id

    # At app startup:
    configure_logging(log_level="INFO", log_dir="logs")
    start_logging_listener()

    # At app shutdown:
    stop_logging_listener()

    # In any module:
    logger = get_logger(__name__)          # routes to SYSTEM log
    rr_logger = get_req_resp_logger()      # routes to REQ_RESP log
    ui_logger = get_ui_logger()            # routes to UI log
"""
import logging
import os
import queue
import sys
import threading
from contextvars import ContextVar
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

# ── Per-request context vars ──────────────────────────────────────────────────

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_request_info: ContextVar[str] = ContextVar("request_info", default="-")


def set_trace_id(value: str) -> None:
    _trace_id.set(value)


def get_trace_id() -> str:
    return _trace_id.get()


def set_request_info(method: str, path: str, client_ip: str = "-") -> None:
    _request_info.set(f"{client_ip} {method} {path}")


def get_request_info() -> str:
    return _request_info.get()


# ── Filters ───────────────────────────────────────────────────────────────────

class ContextFilter(logging.Filter):
    """Injects trace_id and request_info into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        record.request_info = _request_info.get()
        return True


class ErrorOnlyFilter(logging.Filter):
    """Passes only WARNING and above."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


# ── Thread-safe rotating handler ─────────────────────────────────────────────

class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    TimedRotatingFileHandler with an explicit Lock around doRollover() to
    prevent race conditions when multiple threads trigger rotation at the
    same wall-clock second.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollover_lock = threading.Lock()

    def doRollover(self) -> None:
        with self._rollover_lock:
            super().doRollover()


# ── Log formats ───────────────────────────────────────────────────────────────

_FULL_FORMAT = (
    "%(asctime)s | %(levelname)-8s | pid=%(process)d | "
    "%(trace_id)s | %(request_info)s | %(name)s:%(lineno)d | %(message)s"
)
_REQ_RESP_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(trace_id)s | %(request_info)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ── Internal state ────────────────────────────────────────────────────────────

_configured = False
_listeners: list[QueueListener] = []

# Named logger constants
SYSTEM_LOGGER = "gateway.system"
ERROR_LOGGER = "gateway.error"
REQ_RESP_LOGGER = "gateway.req_resp"
UI_LOGGER = "gateway.ui"


# ── File handler factory ──────────────────────────────────────────────────────

def _make_file_handler(
    log_dir: str,
    category: str,
    formatter: logging.Formatter,
    extra_filter: Optional[logging.Filter] = None,
) -> SafeTimedRotatingFileHandler:
    """
    Create a SafeTimedRotatingFileHandler that writes to:
        {log_dir}/YYYY-MM-DD/gateway_{CATEGORY}_{YYYY-MM-DD}_{HH}.log

    The date folder and hourly suffix are baked into the base filename at
    handler-creation time.  A new handler is NOT created each hour — the
    TimedRotatingFileHandler's rollover renames the file automatically and
    appends the hour suffix on rotation.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    hour  = datetime.now().strftime("%H")
    date_dir = Path(log_dir) / today
    date_dir.mkdir(parents=True, exist_ok=True)

    filename = date_dir / f"gateway_{category}_{today}_{hour}.log"
    handler = SafeTimedRotatingFileHandler(
        filename=str(filename),
        when="H",           # rotate every hour
        interval=1,
        backupCount=72,     # keep 72 hourly files (3 days)
        encoding="utf-8",
        delay=False,
    )
    # Suffix so rotated files keep the hour in their name: _2026-05-19_15
    handler.suffix = "%Y-%m-%d_%H"
    handler.setFormatter(formatter)
    if extra_filter:
        handler.addFilter(extra_filter)
    return handler


# ── Public API ────────────────────────────────────────────────────────────────

def configure_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Set up all four log categories.  Safe to call multiple times (idempotent).
    Call start_logging_listener() after this to begin async dispatch.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, log_level.upper(), logging.INFO)
    ctx_filter = ContextFilter()

    full_fmt    = logging.Formatter(_FULL_FORMAT,     datefmt=_DATE_FORMAT)
    rr_fmt      = logging.Formatter(_REQ_RESP_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler (stdout, all levels) ──────────────────────────────────
    stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
    console = logging.StreamHandler(stdout)
    console.setLevel(level)
    console.setFormatter(full_fmt)
    console.addFilter(ctx_filter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)

    # ── SYSTEM logger ─────────────────────────────────────────────────────────
    _setup_async_logger(
        name=SYSTEM_LOGGER,
        level=level,
        handler=_make_file_handler(log_dir, "SYSTEM", full_fmt),
        ctx_filter=ctx_filter,
    )

    # ── ERROR logger (WARNING+ from all loggers) ──────────────────────────────
    _setup_async_logger(
        name=ERROR_LOGGER,
        level=logging.WARNING,
        handler=_make_file_handler(log_dir, "ERROR", full_fmt),
        ctx_filter=ctx_filter,
        extra_filter=ErrorOnlyFilter(),
    )

    # ── REQ_RESP logger ───────────────────────────────────────────────────────
    _setup_async_logger(
        name=REQ_RESP_LOGGER,
        level=level,
        handler=_make_file_handler(log_dir, "REQ_RESP", rr_fmt),
        ctx_filter=ctx_filter,
    )

    # ── UI logger ─────────────────────────────────────────────────────────────
    _setup_async_logger(
        name=UI_LOGGER,
        level=level,
        handler=_make_file_handler(log_dir, "UI", full_fmt),
        ctx_filter=ctx_filter,
    )

    # Wire the error logger as a handler on the root so all WARNING+ records
    # are duplicated into the ERROR log regardless of which logger they originate from.
    err_queue: queue.Queue = logging.getLogger(ERROR_LOGGER).handlers[0].queue  # type: ignore[attr-defined]
    root.addHandler(QueueHandler(err_queue))


def _setup_async_logger(
    name: str,
    level: int,
    handler: logging.Handler,
    ctx_filter: logging.Filter,
    extra_filter: Optional[logging.Filter] = None,
) -> None:
    """Wire a named logger to a QueueHandler backed by a QueueListener."""
    log_queue: queue.Queue = queue.Queue(maxsize=-1)

    queue_handler = QueueHandler(log_queue)
    queue_handler.addFilter(ctx_filter)
    if extra_filter:
        queue_handler.addFilter(extra_filter)

    named_logger = logging.getLogger(name)
    named_logger.setLevel(level)
    named_logger.propagate = False
    named_logger.handlers.clear()
    named_logger.addHandler(queue_handler)

    listener = QueueListener(log_queue, handler, respect_handler_level=True)
    _listeners.append(listener)


def start_logging_listener() -> None:
    """Start all QueueListener background threads.  Call once at app startup."""
    for listener in _listeners:
        listener.start()


def stop_logging_listener() -> None:
    """Flush and stop all QueueListener background threads.  Call at shutdown."""
    for listener in _listeners:
        listener.stop()


# ── Logger accessors ──────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """General-purpose logger — writes to console + SYSTEM log file."""
    return logging.getLogger(name)


def get_req_resp_logger() -> logging.Logger:
    """Request/response logger — writes to REQ_RESP log file."""
    return logging.getLogger(REQ_RESP_LOGGER)


def get_ui_logger() -> logging.Logger:
    """UI/Streamlit logger — writes to UI log file."""
    return logging.getLogger(UI_LOGGER)
