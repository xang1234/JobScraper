"""Rate limiting and request logging middleware for the FastAPI application."""

import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("src.api.access")


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, respecting X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # First IP in the comma-separated list is the original client
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP.

    Uses ``time.monotonic()`` for a clock that can't jump backwards.
    Expired timestamps are pruned on each request for the current IP,
    keeping memory bounded.
    """

    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        # IP -> list of monotonic timestamps
        self._hits: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        ip = get_client_ip(request)
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Prune expired timestamps for this IP
        timestamps = [t for t in self._hits.get(ip, ()) if t > cutoff]

        if not timestamps:
            # Remove stale key so the dict doesn't grow unbounded
            self._hits.pop(ip, None)
        else:
            self._hits[ip] = timestamps

        if len(timestamps) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": f"Rate limit exceeded ({self.requests_per_minute} requests/minute)",
                    }
                },
            )

        timestamps.append(now)
        self._hits[ip] = timestamps
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with method, path, status, duration, and client IP."""

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000

        ip = get_client_ip(request)
        logger.info(
            "%s %s %d %.0fms %s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            ip,
        )
        return response
