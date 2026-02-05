"""
Tests for MCFDatabase class.

These tests verify database operations including:
- Job upsert with history tracking
- Query and search functionality
- Embedding storage and retrieval
- FTS5 full-text search
- Session management
"""

import numpy as np
import pytest
from pathlib import Path

from src.mcf.database import MCFDatabase
from src.mcf.models import Job

from .factories import generate_test_job, generate_test_jobs


class TestDatabaseCreation:
    """Tests for database initialization."""

    def test_creates_database_file(self, temp_dir: Path):
        """Test database file is created."""
        db_path = temp_dir / "test.db"
        db = MCFDatabase(str(db_path))

        assert db_path.exists()
        assert db.db_path == db_path

    def test_creates_parent_directories(self, temp_dir: Path):
        """Test parent directories are created."""
        db_path = temp_dir / "subdir" / "deep" / "test.db"
        db = MCFDatabase(str(db_path))

        assert db_path.parent.exists()
        assert db_path.exists()

    def test_schema_is_created(self, empty_db: MCFDatabase):
        """Test all tables are created."""
        with empty_db._connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t[0] for t in tables}

        expected_tables = {
            "jobs",
            "job_history",
            "scrape_sessions",
            "historical_scrape_progress",
            "fetch_attempts",
            "daemon_state",
            "embeddings",
            "search_analytics",
            "jobs_fts",  # FTS5 virtual table
        }

        assert expected_tables.issubset(table_names)


class TestJobUpsert:
    """Tests for job insert and update operations."""

    def test_insert_new_job(self, empty_db: MCFDatabase, sample_job: Job):
        """Test inserting a new job."""
        is_new, was_updated = empty_db.upsert_job(sample_job)

        assert is_new is True
        assert was_updated is False
        assert empty_db.count_jobs() == 1

    def test_upsert_existing_unchanged(self, empty_db: MCFDatabase, sample_job: Job):
        """Test upserting same job twice without changes."""
        empty_db.upsert_job(sample_job)
        is_new, was_updated = empty_db.upsert_job(sample_job)

        assert is_new is False
        assert was_updated is False
        assert empty_db.count_jobs() == 1

    def test_upsert_detects_changes(self, empty_db: MCFDatabase):
        """Test upsert detects and records changes."""
        job = generate_test_job(title="Original Title", salary_min=5000)
        empty_db.upsert_job(job)

        # Create updated version with same UUID
        updated_job = generate_test_job(
            title="Updated Title",
            salary_min=6000,
            job_uuid=job.uuid,
        )
        is_new, was_updated = empty_db.upsert_job(updated_job)

        assert is_new is False
        assert was_updated is True

        # Verify history was created
        history = empty_db.get_job_history(job.uuid)
        assert len(history) == 1
        assert history[0]["title"] == "Original Title"

    def test_get_job_by_uuid(self, test_db: MCFDatabase):
        """Test retrieving job by UUID."""
        jobs = test_db.search_jobs(limit=1)
        uuid = jobs[0]["uuid"]

        job = test_db.get_job(uuid)

        assert job is not None
        assert job["uuid"] == uuid

    def test_get_nonexistent_job(self, empty_db: MCFDatabase):
        """Test retrieving non-existent job returns None."""
        job = empty_db.get_job("nonexistent-uuid")
        assert job is None

    def test_has_job(self, test_db: MCFDatabase):
        """Test has_job check."""
        jobs = test_db.search_jobs(limit=1)
        uuid = jobs[0]["uuid"]

        assert test_db.has_job(uuid) is True
        assert test_db.has_job("nonexistent-uuid") is False

    def test_get_all_uuids(self, test_db: MCFDatabase):
        """Test getting all job UUIDs."""
        uuids = test_db.get_all_uuids()

        assert len(uuids) == 20  # test_db has 20 jobs
        assert all(isinstance(u, str) for u in uuids)


class TestJobSearch:
    """Tests for job search functionality."""

    def test_search_by_keyword(self, test_db: MCFDatabase):
        """Test keyword search."""
        # Insert a job with known title
        known_job = generate_test_job(title="Unique Quantum Engineer")
        test_db.upsert_job(known_job)

        results = test_db.search_jobs(keyword="Quantum")

        assert len(results) >= 1
        assert any("Quantum" in r["title"] for r in results)

    def test_search_by_company(self, test_db_with_company_groups: MCFDatabase):
        """Test company filter."""
        results = test_db_with_company_groups.search_jobs(company_name="Google")

        assert len(results) == 5  # We created 5 Google jobs
        assert all("Google" in r["company_name"] for r in results)

    def test_search_by_salary_min(self, test_db: MCFDatabase, high_salary_jobs: list[Job]):
        """Test minimum salary filter."""
        for job in high_salary_jobs:
            test_db.upsert_job(job)

        results = test_db.search_jobs(salary_min=10000)

        assert all(r["salary_min"] >= 10000 for r in results if r["salary_min"])

    def test_search_by_salary_max(self, test_db: MCFDatabase, low_salary_jobs: list[Job]):
        """Test maximum salary filter."""
        for job in low_salary_jobs:
            test_db.upsert_job(job)

        results = test_db.search_jobs(salary_max=6000)

        assert all(r["salary_max"] <= 6000 for r in results if r["salary_max"])

    def test_search_with_limit(self, large_test_db: MCFDatabase):
        """Test result limit."""
        results = large_test_db.search_jobs(limit=10)
        assert len(results) == 10

    def test_search_with_offset(self, large_test_db: MCFDatabase):
        """Test pagination offset."""
        first_page = large_test_db.search_jobs(limit=10, offset=0)
        second_page = large_test_db.search_jobs(limit=10, offset=10)

        # Pages should have different jobs
        first_uuids = {j["uuid"] for j in first_page}
        second_uuids = {j["uuid"] for j in second_page}

        assert len(first_uuids & second_uuids) == 0  # No overlap


class TestJobHistory:
    """Tests for job history tracking."""

    def test_history_records_changes(self, empty_db: MCFDatabase):
        """Test that changes are recorded in history."""
        job = generate_test_job(title="Version 1", salary_min=5000)
        empty_db.upsert_job(job)

        # Update multiple times
        for i in range(2, 5):
            updated = generate_test_job(
                title=f"Version {i}",
                salary_min=5000 + (i * 1000),
                job_uuid=job.uuid,
            )
            empty_db.upsert_job(updated)

        history = empty_db.get_job_history(job.uuid)

        assert len(history) == 3  # 3 updates after initial insert
        # Verify all versions are recorded (order may vary with rapid inserts)
        titles = {h["title"] for h in history}
        assert titles == {"Version 1", "Version 2", "Version 3"}

    def test_no_history_for_new_job(self, empty_db: MCFDatabase, sample_job: Job):
        """Test new jobs have no history."""
        empty_db.upsert_job(sample_job)
        history = empty_db.get_job_history(sample_job.uuid)

        assert len(history) == 0


class TestDatabaseStats:
    """Tests for database statistics."""

    def test_get_stats(self, test_db: MCFDatabase):
        """Test stats retrieval."""
        stats = test_db.get_stats()

        assert stats["total_jobs"] == 20
        assert "by_employment_type" in stats
        assert "top_companies" in stats
        assert "salary_stats" in stats

    def test_count_jobs(self, test_db: MCFDatabase):
        """Test job count."""
        assert test_db.count_jobs() == 20

    def test_empty_db_stats(self, empty_db: MCFDatabase):
        """Test stats on empty database."""
        stats = empty_db.get_stats()

        assert stats["total_jobs"] == 0


class TestEmbeddings:
    """Tests for embedding storage and retrieval."""

    def test_upsert_embedding(self, empty_db: MCFDatabase, mock_embedding: np.ndarray):
        """Test inserting an embedding."""
        empty_db.upsert_embedding(
            entity_id="test-uuid",
            entity_type="job",
            embedding=mock_embedding,
        )

        retrieved = empty_db.get_embedding("test-uuid", "job")

        assert retrieved is not None
        assert retrieved.shape == (384,)
        np.testing.assert_array_almost_equal(retrieved, mock_embedding)

    def test_update_embedding(self, empty_db: MCFDatabase):
        """Test updating an existing embedding."""
        original = np.random.randn(384).astype(np.float32)
        updated = np.random.randn(384).astype(np.float32)

        empty_db.upsert_embedding("test-uuid", "job", original)
        empty_db.upsert_embedding("test-uuid", "job", updated)

        retrieved = empty_db.get_embedding("test-uuid", "job")
        np.testing.assert_array_almost_equal(retrieved, updated)

    def test_get_nonexistent_embedding(self, empty_db: MCFDatabase):
        """Test retrieving non-existent embedding returns None."""
        embedding = empty_db.get_embedding("nonexistent", "job")
        assert embedding is None

    def test_get_all_embeddings(self, test_db_with_embeddings: MCFDatabase):
        """Test bulk embedding retrieval."""
        ids, embeddings = test_db_with_embeddings.get_all_embeddings("job")

        assert len(ids) == 20  # test_db has 20 jobs
        assert embeddings.shape == (20, 384)

    def test_batch_upsert_embeddings(
        self, empty_db: MCFDatabase, mock_embeddings_batch: tuple
    ):
        """Test batch embedding insertion."""
        uuids, embeddings = mock_embeddings_batch

        count = empty_db.batch_upsert_embeddings(
            entity_ids=uuids,
            entity_type="job",
            embeddings=embeddings,
        )

        assert count == 20

        # Verify all were inserted
        ids, retrieved = empty_db.get_all_embeddings("job")
        assert len(ids) == 20

    def test_get_embeddings_for_uuids(
        self, test_db_with_embeddings: MCFDatabase
    ):
        """Test retrieving specific embeddings by UUID."""
        # Get some UUIDs from the database
        jobs = test_db_with_embeddings.search_jobs(limit=5)
        uuids = [j["uuid"] for j in jobs]

        embeddings = test_db_with_embeddings.get_embeddings_for_uuids(uuids)

        assert len(embeddings) == 5
        assert all(uuid in embeddings for uuid in uuids)

    def test_embedding_stats(self, test_db_with_embeddings: MCFDatabase):
        """Test embedding statistics."""
        stats = test_db_with_embeddings.get_embedding_stats()

        assert stats["job_embeddings"] == 20
        assert stats["total_jobs"] == 20
        assert stats["coverage_pct"] == 100.0

    def test_delete_embeddings_for_model(
        self, test_db_with_embeddings: MCFDatabase
    ):
        """Test deleting embeddings by model version."""
        # All test embeddings use default model
        count = test_db_with_embeddings.delete_embeddings_for_model(
            "all-MiniLM-L6-v2"
        )

        assert count == 20

        stats = test_db_with_embeddings.get_embedding_stats()
        assert stats["job_embeddings"] == 0


class TestFTS5Search:
    """Tests for full-text search functionality."""

    def test_bm25_search(self, test_db: MCFDatabase):
        """Test BM25 full-text search."""
        # Insert job with unique term
        unique_job = generate_test_job(title="Xylophone Technician")
        test_db.upsert_job(unique_job)

        results = test_db.bm25_search("Xylophone")

        assert len(results) >= 1
        assert results[0][0] == unique_job.uuid  # UUID should match

    def test_bm25_returns_scores(self, test_db: MCFDatabase):
        """Test BM25 returns relevance scores."""
        results = test_db.bm25_search("Engineer")

        assert all(isinstance(r[1], float) for r in results)

    def test_rebuild_fts_index(self, test_db: MCFDatabase):
        """Test FTS index rebuild doesn't error."""
        # Should complete without error
        test_db.rebuild_fts_index()

        # Search should still work
        results = test_db.bm25_search("Engineer")
        assert len(results) >= 0  # May or may not have results


class TestSearchAnalytics:
    """Tests for search analytics logging."""

    def test_log_search(self, empty_db: MCFDatabase):
        """Test logging a search."""
        empty_db.log_search(
            query="data scientist",
            query_type="semantic",
            result_count=10,
            latency_ms=50.5,
            cache_hit=False,
        )

        popular = empty_db.get_popular_queries(days=1)
        assert len(popular) == 1
        assert popular[0]["query"] == "data scientist"

    def test_analytics_summary(self, empty_db: MCFDatabase):
        """Test analytics summary."""
        # Log several searches
        for i in range(5):
            empty_db.log_search(
                query=f"query {i}",
                query_type="semantic",
                result_count=10,
                latency_ms=50.0,
            )

        summary = empty_db.get_analytics_summary(days=1)

        assert summary["total_searches"] == 5
        assert summary["cache_hit_rate"] == 0.0

    def test_latency_percentiles(self, empty_db: MCFDatabase):
        """Test latency percentile calculation."""
        # Log searches with varying latencies
        for latency in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            empty_db.log_search(
                query="test",
                query_type="semantic",
                result_count=10,
                latency_ms=float(latency),
            )

        percentiles = empty_db.get_search_latency_percentiles(days=1)

        assert percentiles["count"] == 10
        # p50 should be around the median (50-60 range with 10 items)
        assert 50 <= percentiles["p50"] <= 60
        assert percentiles["p90"] >= 90


class TestScrapeSessionMethods:
    """Tests for scrape session tracking."""

    def test_create_session(self, empty_db: MCFDatabase):
        """Test creating a scrape session."""
        session_id = empty_db.create_session("data scientist", 1000)

        assert session_id is not None
        assert isinstance(session_id, int)

    def test_update_session(self, empty_db: MCFDatabase):
        """Test updating session progress."""
        session_id = empty_db.create_session("data scientist", 1000)
        empty_db.update_session(session_id, fetched_count=100, current_offset=100)

        session = empty_db.get_incomplete_session("data scientist")
        assert session["fetched_count"] == 100
        assert session["current_offset"] == 100

    def test_complete_session(self, empty_db: MCFDatabase):
        """Test completing a session."""
        session_id = empty_db.create_session("data scientist", 1000)
        empty_db.complete_session(session_id)

        # Should not find incomplete session
        session = empty_db.get_incomplete_session("data scientist")
        assert session is None

    def test_get_all_sessions(self, empty_db: MCFDatabase):
        """Test getting all sessions."""
        empty_db.create_session("query1", 100)
        empty_db.create_session("query2", 200)

        sessions = empty_db.get_all_sessions()
        assert len(sessions) == 2


class TestCompanyAndSkillsMethods:
    """Tests for company and skills helper methods."""

    def test_get_all_unique_skills(self, test_db: MCFDatabase):
        """Test extracting unique skills."""
        skills = test_db.get_all_unique_skills()

        assert len(skills) > 0
        assert isinstance(skills, list)
        assert all(isinstance(s, str) for s in skills)

    def test_get_skill_frequencies(self, test_db: MCFDatabase):
        """Test skill frequency counting."""
        frequencies = test_db.get_skill_frequencies(min_jobs=1, limit=10)

        assert len(frequencies) <= 10
        # Should be sorted by frequency descending
        if len(frequencies) > 1:
            assert frequencies[0][1] >= frequencies[-1][1]

    def test_get_all_unique_companies(self, test_db_with_company_groups: MCFDatabase):
        """Test getting unique companies."""
        companies = test_db_with_company_groups.get_all_unique_companies()

        assert "Google Asia Pacific" in companies
        assert "Meta Platforms Singapore" in companies
        assert "Amazon Web Services" in companies

    def test_get_company_stats(self, test_db_with_company_groups: MCFDatabase):
        """Test company statistics."""
        stats = test_db_with_company_groups.get_company_stats("Google Asia Pacific")

        assert stats["job_count"] == 5
        assert "top_skills" in stats
        assert isinstance(stats["top_skills"], list)

    def test_get_jobs_without_embeddings(self, test_db: MCFDatabase):
        """Test finding jobs without embeddings."""
        jobs = test_db.get_jobs_without_embeddings(limit=10)

        assert len(jobs) <= 10
        assert all("uuid" in j for j in jobs)
        assert all("title" in j for j in jobs)
