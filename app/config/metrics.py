import threading
import time as _time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime as _datetime
from functools import lru_cache
from typing import Deque


@dataclass
class GatewayMetrics:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    requests_total: int = 0
    a2a_tasks_total: int = 0
    session_cache_hits: int = 0
    session_cache_misses: int = 0

    tool_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    llm_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    llm_failovers: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Rolling latency buffer (last 20 requests, milliseconds)
    _latency_buffer: Deque[float] = field(
        default_factory=lambda: deque(maxlen=20), repr=False
    )
    _latency_history: list[float] = field(default_factory=list, repr=False)

    # Error timestamps — used to compute errors-per-minute graph
    _error_timestamps: Deque[float] = field(
        default_factory=lambda: deque(maxlen=500), repr=False
    )

    def record_request(self, latency_ms: float) -> None:
        with self._lock:
            self.requests_total += 1
            self._latency_buffer.append(latency_ms)
            self._latency_history.append(latency_ms)
            if len(self._latency_history) > 100:
                self._latency_history = self._latency_history[-100:]

    def record_tool_call(self, tool_name: str) -> None:
        with self._lock:
            self.tool_calls[tool_name] += 1

    def record_llm_call(self, provider: str) -> None:
        with self._lock:
            self.llm_calls[provider] += 1

    def record_failover(self, from_provider: str) -> None:
        with self._lock:
            self.llm_failovers[from_provider] += 1

    def record_a2a_task(self) -> None:
        with self._lock:
            self.a2a_tasks_total += 1

    def record_session_hit(self) -> None:
        with self._lock:
            self.session_cache_hits += 1

    def record_session_miss(self) -> None:
        with self._lock:
            self.session_cache_misses += 1

    def record_error(self) -> None:
        with self._lock:
            self._error_timestamps.append(_time.time())

    @property
    def avg_latency_ms(self) -> float:
        buf = list(self._latency_buffer)
        return round(sum(buf) / len(buf), 2) if buf else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.session_cache_hits + self.session_cache_misses
        return round(self.session_cache_hits / total * 100, 1) if total else 0.0

    def errors_per_minute(self, last_n_minutes: int = 10) -> dict[str, int]:
        """Returns {HH:MM label: count} for the last N one-minute buckets.

        Takes a snapshot of the timestamp deque under the lock, then does all
        bucketing outside — safe to call both standalone and from snapshot().
        """
        now = _time.time()
        with self._lock:
            timestamps = list(self._error_timestamps)

        buckets: dict[str, int] = {}
        for i in range(last_n_minutes - 1, -1, -1):
            label = _datetime.fromtimestamp(now - i * 60).strftime("%H:%M")
            buckets[label] = 0
        for ts in timestamps:
            if now - ts < last_n_minutes * 60:
                label = _datetime.fromtimestamp(ts).strftime("%H:%M")
                if label in buckets:
                    buckets[label] += 1
        return buckets

    def snapshot(self) -> dict:
        # Compute errors_per_minute BEFORE acquiring the main lock —
        # it has its own internal lock; calling it inside would deadlock.
        errors_per_min = self.errors_per_minute()
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "a2a_tasks_total": self.a2a_tasks_total,
                "session_cache_hits": self.session_cache_hits,
                "session_cache_misses": self.session_cache_misses,
                "cache_hit_rate_pct": self.cache_hit_rate,
                "avg_latency_ms": self.avg_latency_ms,
                "latency_history": list(self._latency_history),
                "tool_calls": dict(self.tool_calls),
                "llm_calls": dict(self.llm_calls),
                "llm_failovers": dict(self.llm_failovers),
                "errors_per_minute": errors_per_min,
            }


@lru_cache(maxsize=1)
def get_metrics() -> GatewayMetrics:
    return GatewayMetrics()
