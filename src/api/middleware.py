"""Rate limiting and request logging middleware for the FastAPI application."""

import logging
import time
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("src.api.access")


def get_client_ip(
    request: Request,
    trusted_proxies: Optional[frozenset[str]] = None,
) -> str:
    """Extract client IP, only trusting X-Forwarded-For from known proxies.

    Args:
        request: The incoming HTTP request.
        trusted_proxies: Set of proxy IPs allowed to set X-Forwarded-For.
            When the direct connection comes from one of these IPs, the
            first address in X-Forwarded-For is returned.  Otherwise the
            header is ignored and request.client.host is used directly.
    """
    direct_ip = request.client.host if request.client else None

    if trusted_proxies and direct_ip in trusted_proxies:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()

    return direct_ip or "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP.

    Uses ``time.monotonic()`` for a clock that can't jump backwards.
    Expired timestamps are pruned on each request for the current IP,
    keeping memory bounded.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        trusted_proxies: Optional[frozenset[str]] = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self.trusted_proxies = trusted_proxies
        # IP -> list of monotonic timestamps
        self._hits: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        ip = get_client_ip(request, self.trusted_proxies)
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

    def __init__(
        self,
        app,
        trusted_proxies: Optional[frozenset[str]] = None,
    ):
        super().__init__(app)
        self.trusted_proxies = trusted_proxies

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000

        ip = get_client_ip(request, self.trusted_proxies)
        logger.info(
            "%s %s %d %.0fms %s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            ip,
        )
        return response
