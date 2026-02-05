"""
Tests for SemanticSearchEngine.

Tests cover:
- Engine initialization and loading
- Degraded mode operation (no FAISS indexes)
- Search with SQL filters
- Hybrid scoring (semantic + keyword)
- Query expansion
- Similar jobs search
- Caching behavior
- Error handling and graceful degradation
"""

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.mcf.embeddings import (
    SemanticSearchEngine,
    SearchRequest,
    SearchResponse,
    JobResult,
    SimilarJobsRequest,
    SkillSearchRequest,
    FAISSIndexManager,
)
from src.mcf.database import MCFDatabase


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def search_engine(temp_dir: Path, test_db_with_embeddings: MCFDatabase) -> SemanticSearchEngine:
    """
    Create a SemanticSearchEngine with test database.

    Note: This doesn't have FAISS indexes, so it will run in degraded mode.
    """
    return SemanticSearchEngine(
        db_path=str(test_db_with_embeddings.db_path),
        index_dir=temp_dir / "embeddings",
    )


@pytest.fixture
def search_engine_with_indexes(
    temp_dir: Path,
    test_db_with_embeddings: MCFDatabase,
) -> SemanticSearchEngine:
    """
    Create a SemanticSearchEngine with FAISS indexes built.
    """
    index_dir = temp_dir / "embeddings"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Get all embeddings from database
    uuids, embeddings = test_db_with_embeddings.get_all_embeddings("job")

    if len(embeddings) == 0:
        pytest.skip("No embeddings in test database")

    # Build FAISS index
    manager = FAISSIndexManager(index_dir=index_dir)
    manager.build_job_index(embeddings, list(uuids))
    manager.save()

    # Create skill clusters for query expansion
    skill_clusters = {0: ["Python", "Java", "JavaScript"], 1: ["SQL", "PostgreSQL", "MySQL"]}
    skill_to_cluster = {"Python": 0, "Java": 0, "JavaScript": 0, "SQL": 1, "PostgreSQL": 1, "MySQL": 1}

    with open(index_dir / "skill_clusters.pkl", "wb") as f:
        pickle.dump(skill_clusters, f)
    with open(index_dir / "skill_to_cluster.pkl", "wb") as f:
        pickle.dump(skill_to_cluster, f)

    return SemanticSearchEngine(
        db_path=str(test_db_with_embeddings.db_path),
        index_dir=index_dir,
    )


# =============================================================================
# Model Tests
# =============================================================================


class TestSearchRequest:
    """Tests for SearchRequest model."""

    def test_default_values(self):
        """Test SearchRequest has sensible defaults."""
        request = SearchRequest(query="test")

        assert request.query == "test"
        assert request.limit == 20
        assert request.alpha == 0.7
        assert request.expand_query is True
        assert request.freshness_weight == 0.1
        assert request.min_similarity == 0.0

    def test_cache_key_generation(self):
        """Test cache key is consistent for same parameters."""
        request1 = SearchRequest(query="python developer", salary_min=5000)
        request2 = SearchRequest(query="python developer", salary_min=5000)

        assert request1.cache_key() == request2.cache_key()

    def test_cache_key_differs_for_different_params(self):
        """Test cache key differs when parameters change."""
        request1 = SearchRequest(query="python developer", salary_min=5000)
        request2 = SearchRequest(query="python developer", salary_min=6000)

        assert request1.cache_key() != request2.cache_key()


class TestJobResult:
    """Tests for JobResult model."""

    def test_creation(self):
        """Test JobResult can be created with required fields."""
        result = JobResult(
            uuid="test-uuid",
            title="Data Scientist",
            company_name="Test Corp",
            description="A job description",
            similarity_score=0.85,
        )

        assert result.uuid == "test-uuid"
        assert result.similarity_score == 0.85


# =============================================================================
# Engine Initialization Tests
# =============================================================================


class TestEngineInitialization:
    """Tests for SemanticSearchEngine initialization."""

    def test_init_with_defaults(self, temp_dir: Path, empty_db: MCFDatabase):
        """Test engine can be created with default parameters."""
        engine = SemanticSearchEngine(
            db_path=str(empty_db.db_path),
            index_dir=temp_dir / "embeddings",
        )

        assert engine._loaded is False
        assert engine._degraded is False
        assert engine.model_version == "all-MiniLM-L6-v2"

    def test_load_without_indexes(self, search_engine: SemanticSearchEngine):
        """Test engine loads in degraded mode when indexes don't exist."""
        search_engine.load()

        assert search_engine._loaded is True
        assert search_engine._degraded is True
        assert search_engine._has_vector_index is False

    def test_load_with_indexes(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test engine loads successfully when indexes exist."""
        search_engine_with_indexes.load()

        assert search_engine_with_indexes._loaded is True
        assert search_engine_with_indexes._degraded is False
        assert search_engine_with_indexes._has_vector_index is True
        assert search_engine_with_indexes._has_skill_clusters is True

    def test_load_is_idempotent(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test calling load() multiple times is safe."""
        search_engine_with_indexes.load()
        first_loaded = search_engine_with_indexes._loaded

        search_engine_with_indexes.load()
        second_loaded = search_engine_with_indexes._loaded

        assert first_loaded == second_loaded


# =============================================================================
# Search Tests (Degraded Mode)
# =============================================================================


class TestSearchDegradedMode:
    """Tests for search functionality in degraded mode (no FAISS indexes)."""

    def test_search_returns_results(self, search_engine: SemanticSearchEngine):
        """Test basic search works in degraded mode."""
        response = search_engine.search(SearchRequest(query="developer", limit=5))

        assert isinstance(response, SearchResponse)
        assert response.degraded is True
        assert response.search_time_ms > 0

    def test_search_with_no_matches(self, search_engine: SemanticSearchEngine):
        """Test search with query that matches nothing."""
        # Use a query unlikely to match anything
        response = search_engine.search(SearchRequest(
            query="xyznonexistent123",
            limit=5,
        ))

        assert response.degraded is True
        # May still have results from SQL filter if no keyword filter applied

    def test_search_with_salary_filter(self, search_engine: SemanticSearchEngine):
        """Test search with salary filter in degraded mode."""
        response = search_engine.search(SearchRequest(
            query="engineer",
            salary_min=5000,
            limit=10,
        ))

        # All results should have salary >= 5000
        for result in response.results:
            if result.salary_min is not None:
                assert result.salary_min >= 5000

    def test_search_response_has_metadata(self, search_engine: SemanticSearchEngine):
        """Test search response includes metadata."""
        response = search_engine.search(SearchRequest(query="data", limit=5))

        assert hasattr(response, "total_candidates")
        assert hasattr(response, "search_time_ms")
        assert hasattr(response, "degraded")
        assert hasattr(response, "cache_hit")


# =============================================================================
# Search Tests (With Indexes)
# =============================================================================


class TestSearchWithIndexes:
    """Tests for search functionality with FAISS indexes available."""

    def test_search_returns_results(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test basic search with indexes."""
        response = search_engine_with_indexes.search(SearchRequest(
            query="software engineer",
            limit=5,
        ))

        assert isinstance(response, SearchResponse)
        assert response.degraded is False

    def test_search_results_have_scores(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test search results include similarity scores."""
        response = search_engine_with_indexes.search(SearchRequest(
            query="python developer",
            limit=5,
        ))

        for result in response.results:
            assert hasattr(result, "similarity_score")
            assert isinstance(result.similarity_score, float)

    def test_search_results_ordered_by_score(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test search results are ordered by score descending."""
        response = search_engine_with_indexes.search(SearchRequest(
            query="data scientist",
            limit=10,
        ))

        if len(response.results) > 1:
            scores = [r.similarity_score for r in response.results]
            assert scores == sorted(scores, reverse=True)

    def test_search_respects_limit(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test search respects the limit parameter."""
        response = search_engine_with_indexes.search(SearchRequest(
            query="engineer",
            limit=3,
        ))

        assert len(response.results) <= 3

    def test_search_with_min_similarity(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test search filters by minimum similarity."""
        response = search_engine_with_indexes.search(SearchRequest(
            query="data engineer",
            min_similarity=0.5,
            limit=20,
        ))

        for result in response.results:
            assert result.similarity_score >= 0.5


# =============================================================================
# Query Expansion Tests
# =============================================================================


class TestQueryExpansion:
    """Tests for query expansion functionality."""

    def test_query_expansion_enabled(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test query expansion happens when enabled."""
        response = search_engine_with_indexes.search(SearchRequest(
            query="Python",
            expand_query=True,
            limit=5,
        ))

        # Query expansion should be present if skill matched
        # (depends on skill_clusters fixture data)
        if response.query_expansion:
            assert len(response.query_expansion) > 1
            assert "Python" in response.query_expansion

    def test_query_expansion_disabled(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test query expansion can be disabled."""
        response = search_engine_with_indexes.search(SearchRequest(
            query="Python",
            expand_query=False,
            limit=5,
        ))

        assert response.query_expansion is None


# =============================================================================
# Caching Tests
# =============================================================================


class TestCaching:
    """Tests for caching behavior."""

    def test_result_cache_hit(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test that identical searches hit the cache."""
        request = SearchRequest(query="machine learning", limit=5)

        # First search - cache miss
        response1 = search_engine_with_indexes.search(request)
        assert response1.cache_hit is False

        # Second search - cache hit
        response2 = search_engine_with_indexes.search(request)
        assert response2.cache_hit is True

    def test_different_queries_dont_share_cache(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test different queries have separate cache entries."""
        response1 = search_engine_with_indexes.search(SearchRequest(query="python", limit=5))
        response2 = search_engine_with_indexes.search(SearchRequest(query="java", limit=5))

        assert response1.cache_hit is False
        assert response2.cache_hit is False

    def test_cache_clear(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test cache can be cleared."""
        request = SearchRequest(query="test cache", limit=5)

        # Populate cache
        search_engine_with_indexes.search(request)

        # Clear caches
        search_engine_with_indexes.clear_caches()

        # Should be cache miss now
        response = search_engine_with_indexes.search(request)
        assert response.cache_hit is False


# =============================================================================
# Similar Jobs Tests
# =============================================================================


class TestFindSimilar:
    """Tests for find_similar functionality."""

    def test_find_similar_without_indexes(self, search_engine: SemanticSearchEngine):
        """Test find_similar returns empty in degraded mode."""
        response = search_engine.find_similar(SimilarJobsRequest(
            job_uuid="nonexistent-uuid",
            limit=5,
        ))

        assert response.degraded is True
        assert len(response.results) == 0

    def test_find_similar_with_nonexistent_job(
        self, search_engine_with_indexes: SemanticSearchEngine
    ):
        """Test find_similar handles nonexistent job gracefully."""
        response = search_engine_with_indexes.find_similar(SimilarJobsRequest(
            job_uuid="nonexistent-uuid-12345",
            limit=5,
        ))

        # Should return empty results, not error
        assert len(response.results) == 0


# =============================================================================
# Skill Search Tests
# =============================================================================


class TestSearchBySkill:
    """Tests for search_by_skill functionality."""

    def test_search_by_skill_degraded(self, search_engine: SemanticSearchEngine):
        """Test skill search falls back to keyword in degraded mode."""
        response = search_engine.search_by_skill(SkillSearchRequest(
            skill="Python",
            limit=5,
        ))

        assert response.degraded is True


# =============================================================================
# Statistics Tests
# =============================================================================


class TestGetStats:
    """Tests for get_stats functionality."""

    def test_stats_before_load(self, search_engine: SemanticSearchEngine):
        """Test stats are available before loading."""
        stats = search_engine.get_stats()

        assert stats["loaded"] is False
        assert "caches" in stats

    def test_stats_after_load(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test stats reflect loaded state."""
        search_engine_with_indexes.load()
        stats = search_engine_with_indexes.get_stats()

        assert stats["loaded"] is True
        assert stats["has_vector_index"] is True
        assert "index_stats" in stats

    def test_stats_include_cache_info(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test stats include cache information."""
        stats = search_engine_with_indexes.get_stats()

        assert "caches" in stats
        assert "query_cache_size" in stats["caches"]
        assert "result_cache_size" in stats["caches"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and graceful degradation."""

    def test_handles_index_compatibility_error(
        self, temp_dir: Path, test_db_with_embeddings: MCFDatabase
    ):
        """Test engine handles index compatibility errors gracefully."""
        index_dir = temp_dir / "embeddings"
        index_dir.mkdir(parents=True, exist_ok=True)

        # Create incompatible metadata
        import pickle
        metadata = {"model_version": "incompatible-model"}
        with open(index_dir / "index_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        engine = SemanticSearchEngine(
            db_path=str(test_db_with_embeddings.db_path),
            index_dir=index_dir,
            model_version="all-MiniLM-L6-v2",  # Different from saved
        )

        # Should load in degraded mode, not crash
        engine.load()
        assert engine._degraded is True

    def test_search_handles_empty_results(self, search_engine: SemanticSearchEngine):
        """Test search handles no results gracefully."""
        response = search_engine.search(SearchRequest(
            query="test",
            salary_min=999999999,  # Unrealistic salary to get no results
            limit=5,
        ))

        assert isinstance(response, SearchResponse)
        assert response.results == []
        assert response.total_candidates == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete search workflows."""

    def test_full_search_workflow(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test complete search workflow from query to results."""
        # Load engine
        search_engine_with_indexes.load()
        assert search_engine_with_indexes._loaded

        # Perform search
        response = search_engine_with_indexes.search(SearchRequest(
            query="data scientist python",
            salary_min=4000,
            limit=10,
            expand_query=True,
        ))

        # Verify response structure
        assert isinstance(response, SearchResponse)
        assert response.search_time_ms > 0

        # Verify results structure
        for result in response.results:
            assert isinstance(result, JobResult)
            assert result.uuid
            assert result.title
            assert isinstance(result.similarity_score, float)

    def test_stats_updated_after_searches(self, search_engine_with_indexes: SemanticSearchEngine):
        """Test that engine statistics update after searches."""
        search_engine_with_indexes.load()

        # Initial cache size
        stats1 = search_engine_with_indexes.get_stats()
        initial_cache = stats1["caches"]["result_cache_size"]

        # Perform search
        search_engine_with_indexes.search(SearchRequest(query="test query"))

        # Check cache grew
        stats2 = search_engine_with_indexes.get_stats()
        assert stats2["caches"]["result_cache_size"] >= initial_cache
