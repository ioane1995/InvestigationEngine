"""Small, dependency-free API helpers: rate limiter, TTL cache, metrics, auth."""
from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict, defaultdict, deque
from typing import Any, Optional

from fastapi import Header, HTTPException, Request

from app.config import Settings


class SlidingWindowLimiter:
    """Per-key sliding window (in-memory; fine for a single free-tier instance)."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window = limit, window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, self.window - (now - q[0])
            q.append(now)
            return True, 0.0


class TTLCache:
    def __init__(self, size: int = 256, ttl_s: int = 600):
        self.size, self.ttl = size, ttl_s
        self._d: "OrderedDict[Any, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            v = self._d.get(key)
            if v and time.monotonic() - v[0] < self.ttl:
                self._d.move_to_end(key)
                self.hits += 1
                return v[1]
            if v:
                del self._d[key]
            self.misses += 1
            return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._d[key] = (time.monotonic(), value)
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


class Metrics:
    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.latencies: deque[float] = deque(maxlen=500)
        self._lock = threading.Lock()

    def inc(self, name: str, v: float = 1.0) -> None:
        with self._lock:
            self.counters[name] += v

    def observe(self, ms: float) -> None:
        with self._lock:
            self.latencies.append(ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = sorted(self.latencies)
            pct = lambda p: round(lat[min(len(lat) - 1, int(p * (len(lat) - 1)))], 1) if lat else None  # noqa: E731
            return {"counters": dict(self.counters), "ask_latency_ms": {"n": len(lat), "p50": pct(0.5), "p95": pct(0.95)}}


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def make_admin_dependency(settings: Settings):
    async def require_admin(x_api_key: Optional[str] = Header(default=None)) -> None:
        if not settings.admin_api_key:
            if settings.is_production:
                raise HTTPException(403, "write endpoints are disabled: ADMIN_API_KEY is not configured")
            return  # open in local development
        if not x_api_key or not secrets.compare_digest(x_api_key, settings.admin_api_key):
            raise HTTPException(401, "missing or invalid X-API-Key")
    return require_admin
