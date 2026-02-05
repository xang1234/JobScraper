"""
Integration tests for the full search pipeline.

These tests verify end-to-end functionality when all components
(database, embeddings, search) work together.

Note: Some tests are marked with @pytest.mark.embedding and will be
skipped until the embedding infrastructure is implemented.
"""

import numpy as np
import pytest

from src.mcf.database import MCFDatabase

from .factories import generate_test_job, generate_similar_jobs


class TestDatabaseEmbeddingIntegration:
    """Integration tests for database + embedding operations."""

    def test_job_insert_with_embedding(self, empty_db: MCFDatabase):
        """Test inserting a job and its embedding together."""
        job = generate_test_job()
        empty_db.upsert_job(job)

        # Generate and store embedding
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        empty_db.upsert_embedding(job.uuid, "job", embedding)

        # Verify both exist
        assert empty_db.has_job(job.uuid)
        retrieved_embedding = empty_db.get_embedding(job.uuid, "job")
        assert retrieved_embedding is not None

    def test_embedding_coverage_tracking(self, test_db: MCFDatabase):
        """Test that embedding coverage is tracked correctly."""
        # Initially no embeddings
        stats = test_db.get_embedding_stats()
        assert stats["coverage_pct"] == 0.0

        # Add embeddings for half the jobs
        jobs = test_db.search_jobs(limit=10)
        for job in jobs:
            embedding = np.random.randn(384).astype(np.float32)
            test_db.upsert_embedding(job["uuid"], "job", embedding)

        stats = test_db.get_embedding_stats()
        assert stats["coverage_pct"] == 50.0  # 10 out of 20 jobs

    def test_find_jobs_without_embeddings(self, test_db: MCFDatabase):
        """Test finding jobs that need embeddings."""
        # Add embeddings for some jobs
        jobs = test_db.search_jobs(limit=5)
        for job in jobs:
            embedding = np.random.randn(384).astype(np.float32)
            test_db.upsert_embedding(job["uuid"], "job", embedding)

        # Should find remaining jobs
        missing = test_db.get_jobs_without_embeddings(limit=100)
        assert len(missing) == 15  # 20 - 5 = 15


class TestFTSIntegration:
    """Integration tests for full-text search."""

    def test_fts_syncs_with_job_insert(self, empty_db: MCFDatabase):
        """Test FTS index updates when jobs are inserted."""
        job = generate_test_job(title="Quantum Computing Specialist")
        empty_db.upsert_job(job)

        # FTS should find the job
        results = empty_db.bm25_search("Quantum")
        assert len(results) == 1
        assert results[0][0] == job.uuid

    def test_fts_syncs_with_job_update(self, empty_db: MCFDatabase):
        """Test FTS index updates when jobs are updated."""
        job = generate_test_job(title="Original Title", salary_min=5000)
        empty_db.upsert_job(job)

        # Search should find original
        results = empty_db.bm25_search("Original")
        assert len(results) == 1

        # Update the job
        updated = generate_test_job(
            title="Updated Unique Title",
            salary_min=6000,
            job_uuid=job.uuid,
        )
        empty_db.upsert_job(updated)

        # Old title should not be found
        results = empty_db.bm25_search("Original")
        assert len(results) == 0

        # New title should be found
        results = empty_db.bm25_search("Updated Unique")
        assert len(results) == 1

    def test_fts_and_sql_combined(self, test_db: MCFDatabase):
        """Test combining FTS search with SQL filters."""
        # Insert a high-salary engineer
        job = generate_test_job(
            title="Specialized Engineer",
            salary_min=15000,
            salary_max=20000,
        )
        test_db.upsert_job(job)

        # FTS finds the job
        fts_results = test_db.bm25_search("Specialized")
        fts_uuids = {r[0] for r in fts_results}

        # SQL also filters
        sql_results = test_db.search_jobs(salary_min=15000)
        sql_uuids = {r["uuid"] for r in sql_results}

        # Our job should be in both
        assert job.uuid in fts_uuids
        assert job.uuid in sql_uuids


class TestSimilaritySearch:
    """Tests for similarity-based searches.

    These tests use mock embeddings to verify the similarity
    calculation logic works correctly.
    """

    def test_embedding_similarity_calculation(self, empty_db: MCFDatabase):
        """Test that similar embeddings are detected."""
        # Create two jobs
        job1 = generate_test_job(title="Python Developer")
        job2 = generate_test_job(title="Python Engineer")
        job3 = generate_test_job(title="Accountant")

        empty_db.upsert_job(job1)
        empty_db.upsert_job(job2)
        empty_db.upsert_job(job3)

        # Create embeddings where job1 and job2 are similar
        base_embedding = np.random.randn(384).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        # job1: base embedding
        empty_db.upsert_embedding(job1.uuid, "job", base_embedding)

        # job2: slightly perturbed (similar)
        similar_embedding = base_embedding + 0.1 * np.random.randn(384).astype(np.float32)
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
        empty_db.upsert_embedding(job2.uuid, "job", similar_embedding)

        # job3: completely different
        different_embedding = np.random.randn(384).astype(np.float32)
        different_embedding = different_embedding / np.linalg.norm(different_embedding)
        empty_db.upsert_embedding(job3.uuid, "job", different_embedding)

        # Get all embeddings and verify
        ids, embeddings = empty_db.get_all_embeddings("job")

        # Calculate similarities
        emb_dict = dict(zip(ids, embeddings))
        sim_1_2 = np.dot(emb_dict[job1.uuid], emb_dict[job2.uuid])
        sim_1_3 = np.dot(emb_dict[job1.uuid], emb_dict[job3.uuid])

        # Similar jobs should have higher similarity
        assert sim_1_2 > sim_1_3


class TestAnalyticsPipeline:
    """Integration tests for analytics tracking."""

    def test_search_logging_and_retrieval(self, empty_db: MCFDatabase):
        """Test full analytics pipeline."""
        # Log several searches
        queries = ["python developer", "data scientist", "ml engineer"]
        for query in queries:
            empty_db.log_search(
                query=query,
                query_type="semantic",
                result_count=10,
                latency_ms=50.0,
            )

        # Log one query multiple times
        for _ in range(5):
            empty_db.log_search(
                query="python developer",
                query_type="semantic",
                result_count=10,
                latency_ms=45.0,
            )

        # Check popular queries
        popular = empty_db.get_popular_queries(days=1)

        # "python developer" should be most popular
        assert popular[0]["query"] == "python developer"
        assert popular[0]["count"] == 6

    def test_latency_tracking_across_queries(self, empty_db: MCFDatabase):
        """Test latency statistics accumulate correctly."""
        # Log searches with varying latencies
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for lat in latencies:
            empty_db.log_search(
                query="test query",
                query_type="semantic",
                result_count=5,
                latency_ms=float(lat),
            )

        percentiles = empty_db.get_search_latency_percentiles(days=1)

        assert percentiles["count"] == 10
        assert 40 <= percentiles["p50"] <= 60  # Median around 55


@pytest.mark.embedding
class TestSemanticSearchPipeline:
    """
    End-to-end tests for the semantic search pipeline.

    These tests are marked with @pytest.mark.embedding and will be
    skipped until EmbeddingGenerator and SemanticSearchEngine are implemented.
    """

    @pytest.mark.skip(reason="Requires EmbeddingGenerator implementation")
    def test_query_to_results_pipeline(self, test_db: MCFDatabase):
        """Test complete flow from query to ranked results."""
        # This test will be enabled once embedding infrastructure exists
        # from src.mcf.embeddings import EmbeddingGenerator, SemanticSearchEngine
        pass

    @pytest.mark.skip(reason="Requires FAISSIndexManager implementation")
    def test_faiss_index_build_and_search(self, test_db_with_embeddings):
        """Test FAISS index building and querying."""
        # from src.mcf.embeddings import FAISSIndexManager
        pass

    @pytest.mark.skip(reason="Requires SemanticSearchEngine implementation")
    def test_hybrid_search_pipeline(self, test_db: MCFDatabase):
        """Test combined FTS + semantic search."""
        # from src.mcf.embeddings import SemanticSearchEngine
        pass
