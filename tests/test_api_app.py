"""Tests for FastAPI application endpoints."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import create_app, get_engine


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class FakeJobResult:
    uuid: str = "job-001"
    title: str = "ML Engineer"
    company_name: str = "TestCo"
    description: str = "A great role"
    salary_min: Optional[int] = 10000
    salary_max: Optional[int] = 15000
    employment_type: str = "Full Time"
    skills: str = "Python, TensorFlow"
    location: str = "Singapore"
    posted_date: Optional[date] = None
    job_url: str = "https://example.com/job"
    similarity_score: float = 0.92


@dataclass
class FakeSearchResponse:
    results: list = field(default_factory=list)
    total_candidates: int = 0
    search_time_ms: float = 42.0
    query_expansion: Optional[list[str]] = None
    degraded: bool = False
    cache_hit: bool = False


@dataclass
class FakeCompanySimilarity:
    company_name: str = "SimilarCo"
    similarity_score: float = 0.85
    job_count: int = 10
    avg_salary: Optional[int] = 12000
    top_skills: list = field(default_factory=lambda: ["Python", "Java"])


def _make_mock_engine(degraded=False, loaded=True):
    """Create a mock SemanticSearchEngine."""
    engine = MagicMock()
    engine._loaded = loaded
    engine._degraded = degraded

    fake_result = FakeJobResult()
    fake_response = FakeSearchResponse(
        results=[fake_result],
        total_candidates=100,
        search_time_ms=42.0,
        query_expansion=["ml", "machine learning"],
    )

    engine.search.return_value = fake_response
    engine.find_similar.return_value = fake_response
    engine.search_by_skill.return_value = fake_response
    engine.find_similar_companies.return_value = [FakeCompanySimilarity()]
    engine.get_stats.return_value = {
        "model_version": "all-MiniLM-L6-v2",
        "embedding_stats": {
            "total_jobs": 50000,
            "job_embeddings": 48000,
            "coverage_pct": 96.0,
            "skill_embeddings": 1200,
            "unique_companies": 3000,
        },
        "index_stats": {
            "index_size_mb": 125.5,
        },
    }
    engine.get_skill_cloud.return_value = {
        "items": [
            {"skill": "Python", "job_count": 5000, "cluster_id": 1},
            {"skill": "Java", "job_count": 3000, "cluster_id": 2},
        ],
        "total_unique_skills": 1200,
    }
    engine.get_related_skills.return_value = {
        "skill": "Python",
        "related": [
            {"skill": "Pandas", "similarity": 0.85, "same_cluster": True},
            {"skill": "NumPy", "similarity": 0.82, "same_cluster": True},
            {"skill": "Django", "similarity": 0.65, "same_cluster": False},
        ],
    }
    engine.db.get_popular_queries.return_value = [
        {"query": "data scientist", "count": 42}
    ]
    engine.db.get_search_latency_percentiles.return_value = {
        "p50": 30.0, "p90": 80.0, "p95": 100.0, "p99": 200.0
    }
    return engine


def _create_test_app(mock_engine) -> FastAPI:
    """Create a FastAPI app with lifespan disabled and engine dependency overridden."""
    app = create_app()
    # Disable lifespan to prevent real SemanticSearchEngine initialization
    app.router.lifespan_context = None
    # Override the get_engine dependency to return our mock
    app.dependency_overrides[get_engine] = lambda: mock_engine
    return app


@pytest.fixture
def mock_engine():
    return _make_mock_engine()


@pytest.fixture
def client(mock_engine):
    """TestClient with mocked search engine (lifespan disabled)."""
    import sys
    app_module = sys.modules["src.api.app"]

    app = _create_test_app(mock_engine)
    # Also set the module-level variable (read by the health endpoint)
    app_module._search_engine = mock_engine
    yield TestClient(app, raise_server_exceptions=False)
    app_module._search_engine = None


@pytest.fixture
def client_no_engine():
    """TestClient with no search engine (simulates engine not initialized)."""
    import sys
    app_module = sys.modules["src.api.app"]

    app = create_app()
    app.router.lifespan_context = None
    app_module._search_engine = None
    yield TestClient(app, raise_server_exceptions=False)
    app_module._search_engine = None


# =============================================================================
# Health Endpoint
# =============================================================================


class TestHealthEndpoint:
    def test_healthy(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["index_loaded"] is True
        assert data["degraded"] is False

    def test_degraded(self):
        import sys
        app_module = sys.modules["src.api.app"]

        engine = _make_mock_engine(degraded=True)
        app = _create_test_app(engine)
        app_module._search_engine = engine
        try:
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["degraded"] is True
        finally:
            app_module._search_engine = None

    def test_no_engine(self, client_no_engine):
        resp = client_no_engine.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["index_loaded"] is False


# =============================================================================
# Search Endpoint
# =============================================================================


class TestSearchEndpoint:
    def test_basic_search(self, client, mock_engine):
        resp = client.post("/api/search", json={"query": "python developer"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "ML Engineer"
        assert data["total_candidates"] == 100
        assert data["search_time_ms"] == 42.0
        mock_engine.search.assert_called_once()

    def test_search_with_filters(self, client, mock_engine):
        resp = client.post("/api/search", json={
            "query": "data scientist",
            "limit": 20,
            "salary_min": 8000,
            "salary_max": 15000,
            "employment_type": "Full Time",
            "alpha": 0.5,
        })
        assert resp.status_code == 200
        call_args = mock_engine.search.call_args[0][0]
        assert call_args.query == "data scientist"
        assert call_args.limit == 20
        assert call_args.salary_min == 8000
        assert call_args.alpha == 0.5

    def test_search_empty_query_rejected(self, client):
        resp = client.post("/api/search", json={"query": ""})
        assert resp.status_code == 422

    def test_search_no_engine(self, client_no_engine):
        resp = client_no_engine.post("/api/search", json={"query": "test"})
        assert resp.status_code == 503

    def test_search_value_error(self, client, mock_engine):
        mock_engine.search.side_effect = ValueError("bad query")
        resp = client.post("/api/search", json={"query": "test"})
        assert resp.status_code == 400
        assert "bad query" in resp.json()["error"]["message"]


# =============================================================================
# Similar Jobs Endpoint
# =============================================================================


class TestSimilarEndpoint:
    def test_similar_jobs(self, client, mock_engine):
        resp = client.post("/api/similar", json={"job_uuid": "abc-123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        mock_engine.find_similar.assert_called_once()

    def test_similar_with_options(self, client, mock_engine):
        resp = client.post("/api/similar", json={
            "job_uuid": "abc-123",
            "limit": 5,
            "exclude_same_company": True,
        })
        assert resp.status_code == 200
        call_args = mock_engine.find_similar.call_args[0][0]
        assert call_args.job_uuid == "abc-123"
        assert call_args.limit == 5
        assert call_args.exclude_same_company is True


# =============================================================================
# Batch Similar Endpoint
# =============================================================================


class TestBatchSimilarEndpoint:
    def test_batch_similar(self, client, mock_engine):
        resp = client.post("/api/similar/batch", json={
            "job_uuids": ["uuid-1", "uuid-2"],
            "limit_per_job": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "uuid-1" in data["results"]
        assert "uuid-2" in data["results"]
        assert data["search_time_ms"] > 0

    def test_batch_calls_engine_per_uuid(self, client, mock_engine):
        resp = client.post("/api/similar/batch", json={
            "job_uuids": ["a", "b", "c"],
        })
        assert resp.status_code == 200
        assert mock_engine.find_similar.call_count == 3


# =============================================================================
# Skill Search Endpoint
# =============================================================================


class TestSkillSearchEndpoint:
    def test_skill_search(self, client, mock_engine):
        resp = client.post("/api/search/skills", json={"skill": "Python"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        mock_engine.search_by_skill.assert_called_once()

    def test_empty_skill_rejected(self, client):
        resp = client.post("/api/search/skills", json={"skill": ""})
        assert resp.status_code == 422


# =============================================================================
# Skill Cloud Endpoint
# =============================================================================


class TestSkillCloudEndpoint:
    def test_skill_cloud(self, client, mock_engine):
        resp = client.get("/api/skills/cloud")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["items"][0]["skill"] == "Python"
        assert data["items"][0]["job_count"] == 5000
        assert data["items"][0]["cluster_id"] == 1
        assert data["total_unique_skills"] == 1200
        mock_engine.get_skill_cloud.assert_called_once()

    def test_skill_cloud_with_params(self, client, mock_engine):
        resp = client.get("/api/skills/cloud?min_jobs=100&limit=50")
        assert resp.status_code == 200
        call_kwargs = mock_engine.get_skill_cloud.call_args
        assert call_kwargs.kwargs["min_jobs"] == 100 or call_kwargs[1]["min_jobs"] == 100

    def test_skill_cloud_invalid_min_jobs(self, client):
        resp = client.get("/api/skills/cloud?min_jobs=0")
        assert resp.status_code == 422

    def test_skill_cloud_limit_too_large(self, client):
        resp = client.get("/api/skills/cloud?limit=501")
        assert resp.status_code == 422

    def test_skill_cloud_no_engine(self, client_no_engine):
        resp = client_no_engine.get("/api/skills/cloud")
        assert resp.status_code == 503


# =============================================================================
# Related Skills Endpoint
# =============================================================================


class TestRelatedSkillsEndpoint:
    def test_related_skills(self, client, mock_engine):
        resp = client.get("/api/skills/related/Python")
        assert resp.status_code == 200
        data = resp.json()
        assert data["skill"] == "Python"
        assert len(data["related"]) == 3
        assert data["related"][0]["skill"] == "Pandas"
        assert data["related"][0]["similarity"] == 0.85
        assert data["related"][0]["same_cluster"] is True
        assert data["related"][2]["same_cluster"] is False

    def test_related_skills_with_k(self, client, mock_engine):
        resp = client.get("/api/skills/related/Python?k=5")
        assert resp.status_code == 200
        call_kwargs = mock_engine.get_related_skills.call_args
        assert call_kwargs.kwargs["k"] == 5 or call_kwargs[1]["k"] == 5

    def test_related_skills_unknown(self, client, mock_engine):
        mock_engine.get_related_skills.return_value = None
        resp = client.get("/api/skills/related/UnknownSkill123")
        assert resp.status_code == 404
        assert "Unknown skill" in resp.json()["error"]["message"]

    def test_related_skills_no_engine(self, client_no_engine):
        resp = client_no_engine.get("/api/skills/related/Python")
        assert resp.status_code == 503

    def test_related_skills_k_too_large(self, client):
        resp = client.get("/api/skills/related/Python?k=51")
        assert resp.status_code == 422


# =============================================================================
# Company Similarity Endpoint
# =============================================================================


class TestCompanySimilarityEndpoint:
    def test_similar_companies(self, client, mock_engine):
        resp = client.post("/api/companies/similar", json={
            "company_name": "Google",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["company_name"] == "SimilarCo"
        assert data[0]["similarity_score"] == 0.85

    def test_empty_company_rejected(self, client):
        resp = client.post("/api/companies/similar", json={
            "company_name": "",
        })
        assert resp.status_code == 422


# =============================================================================
# Stats Endpoint
# =============================================================================


class TestStatsEndpoint:
    def test_stats(self, client, mock_engine):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_jobs"] == 50000
        assert data["jobs_with_embeddings"] == 48000
        assert data["embedding_coverage_pct"] == 96.0
        assert data["model_version"] == "all-MiniLM-L6-v2"


# =============================================================================
# Analytics Endpoints
# =============================================================================


class TestAnalyticsEndpoints:
    def test_popular_queries(self, client, mock_engine):
        resp = client.get("/api/analytics/popular")
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["query"] == "data scientist"

    def test_popular_queries_with_params(self, client, mock_engine):
        resp = client.get("/api/analytics/popular?days=30&limit=5")
        assert resp.status_code == 200
        mock_engine.db.get_popular_queries.assert_called_once_with(days=30, limit=5)

    def test_performance_stats(self, client, mock_engine):
        resp = client.get("/api/analytics/performance")
        assert resp.status_code == 200
        data = resp.json()
        assert "p50" in data
        assert "p99" in data


# =============================================================================
# Error Handling
# =============================================================================


class TestErrorHandling:
    def test_404_for_unknown_route(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404

    def test_405_for_wrong_method(self, client):
        resp = client.get("/api/search")  # Should be POST
        assert resp.status_code == 405

    def test_unhandled_exception_returns_500(self, client, mock_engine):
        mock_engine.search.side_effect = RuntimeError("unexpected")
        resp = client.post("/api/search", json={"query": "test"})
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"]["code"] == "INTERNAL_ERROR"

    def test_503_when_engine_not_ready(self, client_no_engine):
        resp = client_no_engine.post("/api/search", json={"query": "test"})
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"]["code"] == "SERVICE_UNAVAILABLE"


# =============================================================================
# CORS
# =============================================================================


class TestCORS:
    def test_cors_headers_present(self, client):
        resp = client.options(
            "/api/search",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_disallowed_origin(self, client):
        resp = client.options(
            "/api/search",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert resp.headers.get("access-control-allow-origin") is None
