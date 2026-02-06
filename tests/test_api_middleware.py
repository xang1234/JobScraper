"""Tests for rate limiting and request logging middleware."""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    get_client_ip,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_test_app(rate_limit: int | None = None) -> FastAPI:
    """Minimal FastAPI app for isolated middleware testing."""
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"pong": True}

    @app.get("/slow")
    async def slow():
        return {"slow": True}

    if rate_limit is not None:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit)

    app.add_middleware(RequestLoggingMiddleware)
    return app


def _make_request(headers: dict | None = None, client_host: str = "127.0.0.1"):
    """Create a fake Starlette Request for unit-testing get_client_ip."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    if client_host:
        scope["client"] = (client_host, 12345)
    else:
        scope["client"] = None
    return Request(scope)


# =============================================================================
# TestGetClientIp
# =============================================================================


class TestGetClientIp:
    def test_direct_connection(self):
        req = _make_request(client_host="192.168.1.5")
        assert get_client_ip(req) == "192.168.1.5"

    def test_x_forwarded_for_single(self):
        req = _make_request(headers={"X-Forwarded-For": "10.0.0.1"})
        assert get_client_ip(req) == "10.0.0.1"

    def test_x_forwarded_for_chain(self):
        req = _make_request(headers={"X-Forwarded-For": "203.0.113.50, 70.41.3.18, 150.172.238.178"})
        assert get_client_ip(req) == "203.0.113.50"

    def test_no_client_at_all(self):
        req = _make_request(headers={}, client_host=None)
        assert get_client_ip(req) == "unknown"

    def test_forwarded_takes_precedence_over_client(self):
        req = _make_request(
            headers={"X-Forwarded-For": "1.2.3.4"},
            client_host="127.0.0.1",
        )
        assert get_client_ip(req) == "1.2.3.4"


# =============================================================================
# TestRateLimitMiddleware
# =============================================================================


class TestRateLimitMiddleware:
    def test_under_limit_succeeds(self):
        app = _make_test_app(rate_limit=5)
        client = TestClient(app, raise_server_exceptions=False)
        for _ in range(5):
            resp = client.get("/ping")
            assert resp.status_code == 200

    def test_over_limit_returns_429(self):
        app = _make_test_app(rate_limit=3)
        client = TestClient(app, raise_server_exceptions=False)
        for _ in range(3):
            resp = client.get("/ping")
            assert resp.status_code == 200
        resp = client.get("/ping")
        assert resp.status_code == 429

    def test_429_response_format(self):
        app = _make_test_app(rate_limit=1)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/ping")
        resp = client.get("/ping")
        assert resp.status_code == 429
        data = resp.json()
        assert data["error"]["code"] == "RATE_LIMITED"
        assert "1 requests/minute" in data["error"]["message"]

    def test_different_ips_tracked_separately(self):
        app = _make_test_app(rate_limit=2)
        client = TestClient(app, raise_server_exceptions=False)

        # IP A: 2 requests (at limit)
        for _ in range(2):
            resp = client.get("/ping", headers={"X-Forwarded-For": "1.1.1.1"})
            assert resp.status_code == 200

        # IP A: blocked
        resp = client.get("/ping", headers={"X-Forwarded-For": "1.1.1.1"})
        assert resp.status_code == 429

        # IP B: still allowed
        resp = client.get("/ping", headers={"X-Forwarded-For": "2.2.2.2"})
        assert resp.status_code == 200

    def test_window_expiry_allows_new_requests(self):
        """After the sliding window passes, requests should be allowed again."""
        app = _make_test_app(rate_limit=2)
        client = TestClient(app, raise_server_exceptions=False)

        # Fill the window
        client.get("/ping")
        client.get("/ping")
        resp = client.get("/ping")
        assert resp.status_code == 429

        # Fast-forward monotonic time past the 60s window
        with patch("src.api.middleware.time") as mock_time:
            # Simulate that 61 seconds have passed
            future = time.monotonic() + 61
            mock_time.monotonic.return_value = future
            resp = client.get("/ping")
            assert resp.status_code == 200


# =============================================================================
# TestRequestLoggingMiddleware
# =============================================================================


class TestRequestLoggingMiddleware:
    def test_logs_request(self, caplog):
        app = _make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        with caplog.at_level(logging.INFO, logger="src.api.access"):
            client.get("/ping")
        assert any("GET /ping 200" in rec.message for rec in caplog.records)

    def test_logs_duration(self, caplog):
        app = _make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        with caplog.at_level(logging.INFO, logger="src.api.access"):
            client.get("/ping")
        # Should contain a duration like "0ms" or "1ms"
        assert any("ms" in rec.message for rec in caplog.records)

    def test_logs_client_ip(self, caplog):
        app = _make_test_app()
        client = TestClient(app, raise_server_exceptions=False)
        with caplog.at_level(logging.INFO, logger="src.api.access"):
            client.get("/ping", headers={"X-Forwarded-For": "5.6.7.8"})
        assert any("5.6.7.8" in rec.message for rec in caplog.records)

    def test_logs_429_from_rate_limiter(self, caplog):
        app = _make_test_app(rate_limit=1)
        client = TestClient(app, raise_server_exceptions=False)
        with caplog.at_level(logging.INFO, logger="src.api.access"):
            client.get("/ping")  # OK
            client.get("/ping")  # 429
        assert any("429" in rec.message for rec in caplog.records)


# =============================================================================
# TestMiddlewareIntegration
# =============================================================================


class TestMiddlewareIntegration:
    """End-to-end tests using the real create_app factory."""

    def _make_client(self, rate_limit_rpm: int = 100) -> TestClient:
        import sys
        from src.api.app import create_app, get_engine

        app = create_app(rate_limit_rpm=rate_limit_rpm)
        app.router.lifespan_context = None

        # Mock search engine for the health endpoint
        engine = MagicMock()
        engine._loaded = True
        engine._degraded = False
        app.dependency_overrides[get_engine] = lambda: engine
        sys.modules["src.api.app"]._search_engine = engine

        return TestClient(app, raise_server_exceptions=False)

    def test_rate_limit_with_real_app(self):
        client = self._make_client(rate_limit_rpm=3)
        for _ in range(3):
            resp = client.get("/health")
            assert resp.status_code == 200
        resp = client.get("/health")
        assert resp.status_code == 429

    def test_rate_limit_disabled(self):
        client = self._make_client(rate_limit_rpm=0)
        # Should never get rate limited
        for _ in range(200):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_rate_limit_stored_on_app_state(self):
        from src.api.app import create_app
        app = create_app(rate_limit_rpm=42)
        assert app.state.rate_limit_rpm == 42

    def test_logging_with_real_app(self, caplog):
        client = self._make_client()
        with caplog.at_level(logging.INFO, logger="src.api.access"):
            client.get("/health")
        assert any("GET /health 200" in rec.message for rec in caplog.records)

    def test_cors_on_429(self):
        """429 responses should include CORS headers for browser clients."""
        client = self._make_client(rate_limit_rpm=1)
        client.get("/health")
        resp = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert resp.status_code == 429
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
