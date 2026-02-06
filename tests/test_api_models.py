"""Tests for API request/response models."""

from datetime import date

import pytest
from pydantic import ValidationError

from src.api.models import (
    CompanySimilarity,
    CompanySimilarityRequest,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    JobResult,
    RelatedSkill,
    RelatedSkillsResponse,
    SearchRequest,
    SearchResponse,
    SimilarBatchRequest,
    SimilarBatchResponse,
    SimilarJobsRequest,
    SkillCloudItem,
    SkillCloudResponse,
    SkillSearchRequest,
    StatsResponse,
)
from src.mcf.embeddings.models import (
    CompanySimilarity as InternalCompanySimilarity,
    JobResult as InternalJobResult,
    SearchResponse as InternalSearchResponse,
)


# =============================================================================
# SearchRequest Tests
# =============================================================================


class TestSearchRequest:
    def test_minimal_valid(self):
        req = SearchRequest(query="test")
        assert req.query == "test"
        assert req.limit == 10
        assert req.alpha == 0.7
        assert req.freshness_weight == 0.1
        assert req.expand_query is True
        assert req.min_similarity == 0.3

    def test_full_request(self):
        req = SearchRequest(
            query="machine learning",
            limit=50,
            salary_min=8000,
            salary_max=15000,
            employment_type="Full Time",
            region="Central",
            company="Google",
            alpha=0.5,
            freshness_weight=0.2,
            expand_query=False,
            min_similarity=0.5,
        )
        assert req.salary_min == 8000
        assert req.salary_max == 15000
        assert req.company == "Google"

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_query_too_long(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="x" * 501)

    def test_limit_bounds(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=101)

    def test_salary_max_less_than_min(self):
        with pytest.raises(ValidationError, match="salary_max must be >= salary_min"):
            SearchRequest(query="test", salary_min=10000, salary_max=5000)

    def test_salary_max_equal_to_min(self):
        req = SearchRequest(query="test", salary_min=10000, salary_max=10000)
        assert req.salary_max == 10000

    def test_salary_max_without_min_ok(self):
        req = SearchRequest(query="test", salary_max=5000)
        assert req.salary_max == 5000

    def test_negative_salary_rejected(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", salary_min=-1)

    def test_alpha_out_of_range(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", alpha=1.5)
        with pytest.raises(ValidationError):
            SearchRequest(query="test", alpha=-0.1)

    def test_to_internal(self):
        req = SearchRequest(
            query="data engineer",
            limit=20,
            salary_min=8000,
            alpha=0.6,
        )
        internal = req.to_internal()
        assert internal.query == "data engineer"
        assert internal.limit == 20
        assert internal.salary_min == 8000
        assert internal.alpha == 0.6

    def test_json_schema_has_examples(self):
        schema = SearchRequest.model_json_schema()
        assert "examples" in schema


# =============================================================================
# SimilarJobsRequest Tests
# =============================================================================


class TestSimilarJobsRequest:
    def test_valid(self):
        req = SimilarJobsRequest(job_uuid="abc-123")
        assert req.limit == 10
        assert req.exclude_same_company is False

    def test_to_internal(self):
        req = SimilarJobsRequest(
            job_uuid="abc-123", limit=5, exclude_same_company=True
        )
        internal = req.to_internal()
        assert internal.job_uuid == "abc-123"
        assert internal.limit == 5
        assert internal.exclude_same_company is True


# =============================================================================
# SimilarBatchRequest Tests
# =============================================================================


class TestSimilarBatchRequest:
    def test_valid(self):
        req = SimilarBatchRequest(job_uuids=["a", "b", "c"])
        assert req.limit_per_job == 5
        assert req.exclude_same_company is False

    def test_empty_uuids_rejected(self):
        with pytest.raises(ValidationError):
            SimilarBatchRequest(job_uuids=[])

    def test_too_many_uuids_rejected(self):
        with pytest.raises(ValidationError):
            SimilarBatchRequest(job_uuids=["x"] * 51)

    def test_max_uuids_accepted(self):
        req = SimilarBatchRequest(job_uuids=["x"] * 50)
        assert len(req.job_uuids) == 50


# =============================================================================
# SkillSearchRequest Tests
# =============================================================================


class TestSkillSearchRequest:
    def test_valid(self):
        req = SkillSearchRequest(skill="Python")
        assert req.limit == 20

    def test_empty_skill_rejected(self):
        with pytest.raises(ValidationError):
            SkillSearchRequest(skill="")

    def test_salary_validation(self):
        with pytest.raises(ValidationError, match="salary_max must be >= salary_min"):
            SkillSearchRequest(skill="Python", salary_min=10000, salary_max=5000)

    def test_to_internal(self):
        req = SkillSearchRequest(skill="Python", limit=30)
        internal = req.to_internal()
        assert internal.skill == "Python"
        assert internal.limit == 30


# =============================================================================
# CompanySimilarityRequest Tests
# =============================================================================


class TestCompanySimilarityRequest:
    def test_valid(self):
        req = CompanySimilarityRequest(company_name="Google")
        assert req.limit == 10

    def test_empty_company_rejected(self):
        with pytest.raises(ValidationError):
            CompanySimilarityRequest(company_name="")

    def test_to_internal(self):
        req = CompanySimilarityRequest(company_name="Google", limit=5)
        internal = req.to_internal()
        assert internal.company_name == "Google"
        assert internal.limit == 5


# =============================================================================
# Response Model Tests
# =============================================================================


class TestJobResult:
    def test_from_internal(self):
        internal = InternalJobResult(
            uuid="abc-123",
            title="ML Engineer",
            company_name="Google",
            description="Great role",
            salary_min=12000,
            salary_max=18000,
            employment_type="Full Time",
            skills="Python, TensorFlow",
            location="Singapore",
            posted_date=date(2024, 1, 15),
            job_url="https://example.com/job",
            similarity_score=0.923,
        )
        result = JobResult.from_internal(internal)
        assert result.uuid == "abc-123"
        assert result.title == "ML Engineer"
        assert result.similarity_score == 0.923
        assert result.posted_date == date(2024, 1, 15)
        # Extra API fields not set by from_internal
        assert result.bm25_score is None
        assert result.freshness_score is None

    def test_optional_fields(self):
        result = JobResult(
            uuid="x",
            title="Job",
            description="Desc",
            similarity_score=0.5,
        )
        assert result.company_name is None
        assert result.salary_min is None


class TestSearchResponse:
    def test_from_internal(self):
        internal = InternalSearchResponse(
            results=[
                InternalJobResult(
                    uuid="abc",
                    title="Engineer",
                    company_name="Co",
                    description="Desc",
                    similarity_score=0.9,
                )
            ],
            total_candidates=100,
            search_time_ms=42.5,
            query_expansion=["ml", "machine learning"],
            degraded=False,
            cache_hit=True,
        )
        resp = SearchResponse.from_internal(internal)
        assert len(resp.results) == 1
        assert resp.results[0].uuid == "abc"
        assert resp.total_candidates == 100
        assert resp.search_time_ms == 42.5
        assert resp.query_expansion == ["ml", "machine learning"]
        assert resp.cache_hit is True

    def test_defaults(self):
        resp = SearchResponse(
            results=[], total_candidates=0, search_time_ms=1.0
        )
        assert resp.degraded is False
        assert resp.cache_hit is False
        assert resp.query_expansion is None


class TestSimilarBatchResponse:
    def test_structure(self):
        resp = SimilarBatchResponse(
            results={
                "uuid-1": [
                    JobResult(
                        uuid="x",
                        title="Job",
                        description="D",
                        similarity_score=0.8,
                    )
                ],
                "uuid-2": [],
            },
            search_time_ms=123.4,
        )
        assert len(resp.results["uuid-1"]) == 1
        assert resp.results["uuid-2"] == []


class TestCompanySimilarity:
    def test_from_internal(self):
        internal = InternalCompanySimilarity(
            company_name="Google",
            similarity_score=0.85,
            job_count=42,
            avg_salary=15000,
            top_skills=["Python", "Java"],
        )
        result = CompanySimilarity.from_internal(internal)
        assert result.company_name == "Google"
        assert result.job_count == 42
        assert result.top_skills == ["Python", "Java"]


class TestStatsResponse:
    def test_valid(self):
        resp = StatsResponse(
            total_jobs=50000,
            jobs_with_embeddings=48000,
            embedding_coverage_pct=96.0,
            unique_skills=1200,
            unique_companies=3000,
            index_size_mb=125.5,
            model_version="all-MiniLM-L6-v2",
        )
        assert resp.embedding_coverage_pct == 96.0


class TestHealthResponse:
    def test_healthy(self):
        resp = HealthResponse(status="healthy", index_loaded=True, degraded=False)
        assert resp.status == "healthy"

    def test_degraded(self):
        resp = HealthResponse(status="degraded", index_loaded=False, degraded=True)
        assert resp.degraded is True


# =============================================================================
# Skill Feature Model Tests
# =============================================================================


class TestSkillCloudItem:
    def test_valid(self):
        item = SkillCloudItem(skill="Python", job_count=5000, cluster_id=1)
        assert item.skill == "Python"
        assert item.job_count == 5000
        assert item.cluster_id == 1

    def test_cluster_id_optional(self):
        item = SkillCloudItem(skill="Python", job_count=5000)
        assert item.cluster_id is None


class TestSkillCloudResponse:
    def test_valid(self):
        resp = SkillCloudResponse(
            items=[
                SkillCloudItem(skill="Python", job_count=5000, cluster_id=1),
                SkillCloudItem(skill="Java", job_count=3000),
            ],
            total_unique_skills=1200,
        )
        assert len(resp.items) == 2
        assert resp.total_unique_skills == 1200

    def test_empty_items(self):
        resp = SkillCloudResponse(items=[], total_unique_skills=0)
        assert resp.items == []


class TestRelatedSkill:
    def test_valid(self):
        skill = RelatedSkill(skill="Pandas", similarity=0.85, same_cluster=True)
        assert skill.skill == "Pandas"
        assert skill.similarity == 0.85
        assert skill.same_cluster is True


class TestRelatedSkillsResponse:
    def test_valid(self):
        resp = RelatedSkillsResponse(
            skill="Python",
            related=[
                RelatedSkill(skill="Pandas", similarity=0.85, same_cluster=True),
                RelatedSkill(skill="Django", similarity=0.65, same_cluster=False),
            ],
        )
        assert resp.skill == "Python"
        assert len(resp.related) == 2

    def test_empty_related(self):
        resp = RelatedSkillsResponse(skill="ObscureSkill", related=[])
        assert resp.related == []


# =============================================================================
# Error Model Tests
# =============================================================================


class TestErrorResponse:
    def test_structure(self):
        resp = ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="salary_min must be positive",
                details={"field": "salary_min"},
            )
        )
        assert resp.error.code == "VALIDATION_ERROR"
        assert resp.error.details["field"] == "salary_min"

    def test_without_details(self):
        resp = ErrorResponse(
            error=ErrorDetail(code="NOT_FOUND", message="Job not found")
        )
        assert resp.error.details is None
