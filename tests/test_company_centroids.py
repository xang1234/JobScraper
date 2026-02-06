"""
Tests for company multi-centroid integration.

Tests cover the full pipeline:
- Database: get_all_companies()
- Generator: generate_company_centroids_from_db()
- Index manager: get_company_centroids(), has_company_index()
- Search engine: find_similar_companies() with multi-centroid matching
- Degraded mode: fallback when company index unavailable
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.mcf.database import MCFDatabase
from tests.conftest import requires_faiss
from tests.factories import generate_company_job_set


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def db_with_companies(temp_dir):
    """
    Database with 3 companies and embeddings for all jobs.

    Google: 5 jobs, Meta: 5 jobs, Amazon: 5 jobs.
    Each job gets a random embedding stored in the embeddings table.
    """
    db_path = temp_dir / "test.db"
    db = MCFDatabase(str(db_path))

    companies = [
        "Google Asia Pacific",
        "Meta Platforms Singapore",
        "Amazon Web Services",
    ]

    np.random.seed(42)
    for company in companies:
        jobs = generate_company_job_set(company, n=5)
        for job in jobs:
            db.upsert_job(job)

            # Store a random embedding for each job
            embedding = np.random.randn(384).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            db.upsert_embedding(job.uuid, "job", embedding)

    return db


@pytest.fixture
def company_centroids():
    """Generate deterministic company centroids for testing."""
    np.random.seed(789)

    def make_centroids(n):
        centroids = np.random.randn(n, 384).astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        return [c for c in centroids / norms]

    return {
        "Google Asia Pacific": make_centroids(3),
        "Meta Platforms Singapore": make_centroids(2),
        "Amazon Web Services": make_centroids(1),
    }


# =============================================================================
# Database Tests
# =============================================================================


class TestGetAllCompanies:
    """Tests for MCFDatabase.get_all_companies()."""

    def test_returns_distinct_companies(self, db_with_companies):
        """get_all_companies returns all distinct company names."""
        companies = db_with_companies.get_all_companies()

        assert len(companies) == 3
        assert "Google Asia Pacific" in companies
        assert "Meta Platforms Singapore" in companies
        assert "Amazon Web Services" in companies

    def test_returns_sorted(self, db_with_companies):
        """Companies are returned in alphabetical order."""
        companies = db_with_companies.get_all_companies()

        assert companies == sorted(companies)

    def test_empty_db(self, temp_dir):
        """Returns empty list for database with no jobs."""
        db = MCFDatabase(str(temp_dir / "empty.db"))
        companies = db.get_all_companies()

        assert companies == []

    def test_excludes_null_companies(self, temp_dir):
        """Null/empty company names are excluded."""
        db = MCFDatabase(str(temp_dir / "test.db"))

        # Insert a job with a company
        from tests.factories import generate_test_job
        job = generate_test_job(company_name="Real Company")
        db.upsert_job(job)

        companies = db.get_all_companies()
        assert "Real Company" in companies
        # No null/empty entries
        assert all(c for c in companies)


# =============================================================================
# Generator Tests
# =============================================================================


class TestGenerateCompanyCentroidsFromDb:
    """Tests for EmbeddingGenerator.generate_company_centroids_from_db()."""

    def test_produces_centroids(self, db_with_companies):
        """Generates centroids for each company from stored embeddings."""
        from src.mcf.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator()
        centroids = generator.generate_company_centroids_from_db(db_with_companies)

        assert len(centroids) == 3
        for company, company_centroids in centroids.items():
            assert len(company_centroids) >= 1
            for c in company_centroids:
                assert c.shape == (384,)
                # Centroids should be normalized
                assert abs(np.linalg.norm(c) - 1.0) < 1e-5

    def test_small_companies_get_single_centroid(self, db_with_companies):
        """Companies with < 10 jobs get a single mean centroid."""
        from src.mcf.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator()
        centroids = generator.generate_company_centroids_from_db(db_with_companies)

        # Each company has 5 jobs (< 10), so single centroid each
        for company, company_centroids in centroids.items():
            assert len(company_centroids) == 1

    def test_large_company_gets_multiple_centroids(self, temp_dir):
        """Companies with >= 10 jobs get K-means multi-centroids."""
        from src.mcf.embeddings import EmbeddingGenerator

        db = MCFDatabase(str(temp_dir / "large.db"))

        np.random.seed(42)
        jobs = generate_company_job_set("Big Corp", n=15)
        for job in jobs:
            db.upsert_job(job)
            embedding = np.random.randn(384).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            db.upsert_embedding(job.uuid, "job", embedding)

        generator = EmbeddingGenerator()
        centroids = generator.generate_company_centroids_from_db(db, k_centroids=3)

        assert "Big Corp" in centroids
        # 15 jobs with k_centroids=3 -> min(3, 15//3) = 3 centroids
        assert len(centroids["Big Corp"]) == 3

    def test_empty_db_returns_empty(self, temp_dir):
        """Returns empty dict for database with no jobs."""
        from src.mcf.embeddings import EmbeddingGenerator

        db = MCFDatabase(str(temp_dir / "empty.db"))
        generator = EmbeddingGenerator()
        centroids = generator.generate_company_centroids_from_db(db)

        assert centroids == {}

    def test_company_without_embeddings_skipped(self, temp_dir):
        """Companies with no stored embeddings are skipped."""
        from src.mcf.embeddings import EmbeddingGenerator

        db = MCFDatabase(str(temp_dir / "test.db"))

        # Insert jobs but no embeddings
        jobs = generate_company_job_set("No Embeddings Corp", n=5)
        for job in jobs:
            db.upsert_job(job)

        generator = EmbeddingGenerator()
        centroids = generator.generate_company_centroids_from_db(db)

        assert "No Embeddings Corp" not in centroids


# =============================================================================
# Index Manager Tests
# =============================================================================


@requires_faiss
class TestGetCompanyCentroids:
    """Tests for FAISSIndexManager.get_company_centroids()."""

    def test_returns_centroids_for_known_company(self, temp_dir, company_centroids):
        """Returns centroid array for a company in the index."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_dir)
        manager.build_company_index(company_centroids)

        result = manager.get_company_centroids("Google Asia Pacific")

        assert result is not None
        assert result.shape == (3, 384)  # 3 centroids

    def test_returns_none_for_unknown_company(self, temp_dir, company_centroids):
        """Returns None for company not in the index."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_dir)
        manager.build_company_index(company_centroids)

        result = manager.get_company_centroids("Unknown Corp")

        assert result is None

    def test_returns_none_without_index(self, temp_dir):
        """Returns None when no company index is built."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_dir)

        result = manager.get_company_centroids("Google Asia Pacific")

        assert result is None

    def test_single_centroid_company(self, temp_dir, company_centroids):
        """Returns 1-centroid array for small companies."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_dir)
        manager.build_company_index(company_centroids)

        result = manager.get_company_centroids("Amazon Web Services")

        assert result is not None
        assert result.shape == (1, 384)


@requires_faiss
class TestHasCompanyIndex:
    """Tests for FAISSIndexManager.has_company_index()."""

    def test_true_when_built(self, temp_dir, company_centroids):
        """Returns True after building company index."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_dir)
        manager.build_company_index(company_centroids)

        assert manager.has_company_index() is True

    def test_false_when_not_built(self, temp_dir):
        """Returns False when no company index exists."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_dir)

        assert manager.has_company_index() is False

    def test_false_when_empty(self, temp_dir):
        """Returns False after building with empty centroids."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_dir)
        manager.build_company_index({})

        assert manager.has_company_index() is False


# =============================================================================
# Search Engine Tests
# =============================================================================


@requires_faiss
class TestFindSimilarCompaniesMultiCentroid:
    """Tests for SemanticSearchEngine.find_similar_companies() with company index."""

    def test_uses_company_index_when_available(
        self, temp_dir, db_with_companies, company_centroids
    ):
        """
        When company index is available, find_similar_companies uses
        multi-centroid matching instead of on-the-fly computation.
        """
        from src.mcf.embeddings import (
            FAISSIndexManager,
            SemanticSearchEngine,
            CompanySimilarityRequest,
        )

        # Build indexes
        index_dir = temp_dir / "indexes"
        index_dir.mkdir()

        # Build job index from stored embeddings
        job_uuids, job_embeddings = db_with_companies.get_all_embeddings("job")
        manager = FAISSIndexManager(index_dir=index_dir)
        manager.build_job_index(job_embeddings, job_uuids)
        manager.build_company_index(company_centroids)
        manager.save()

        # Create engine and load
        engine = SemanticSearchEngine(
            db_path=str(db_with_companies.db_path),
            index_dir=index_dir,
        )
        engine.load()

        request = CompanySimilarityRequest(
            company_name="Google Asia Pacific",
            limit=5,
        )

        results = engine.find_similar_companies(request)

        # Should return results for other companies
        assert len(results) > 0
        # Source company should not be in results
        result_names = [r.company_name for r in results]
        assert "Google Asia Pacific" not in result_names
        # Results should have valid scores
        for r in results:
            assert r.similarity_score > -1.0  # Cosine sim range
            assert r.job_count > 0

    def test_excludes_source_company(
        self, temp_dir, db_with_companies, company_centroids
    ):
        """Source company is excluded from results."""
        from src.mcf.embeddings import (
            FAISSIndexManager,
            SemanticSearchEngine,
            CompanySimilarityRequest,
        )

        index_dir = temp_dir / "indexes"
        index_dir.mkdir()

        job_uuids, job_embeddings = db_with_companies.get_all_embeddings("job")
        manager = FAISSIndexManager(index_dir=index_dir)
        manager.build_job_index(job_embeddings, job_uuids)
        manager.build_company_index(company_centroids)
        manager.save()

        engine = SemanticSearchEngine(
            db_path=str(db_with_companies.db_path),
            index_dir=index_dir,
        )
        engine.load()

        for company in ["Google Asia Pacific", "Meta Platforms Singapore", "Amazon Web Services"]:
            results = engine.find_similar_companies(
                CompanySimilarityRequest(company_name=company, limit=10)
            )
            result_names = [r.company_name for r in results]
            assert company not in result_names


@requires_faiss
class TestFindSimilarCompaniesDegraded:
    """Tests for fallback when company index is not available."""

    def test_falls_back_to_job_index(self, temp_dir, db_with_companies):
        """
        When company index is not built, falls back to on-the-fly
        centroid computation via the jobs index.
        """
        from src.mcf.embeddings import (
            FAISSIndexManager,
            SemanticSearchEngine,
            CompanySimilarityRequest,
        )

        index_dir = temp_dir / "indexes"
        index_dir.mkdir()

        # Build ONLY job index (no company index)
        job_uuids, job_embeddings = db_with_companies.get_all_embeddings("job")
        manager = FAISSIndexManager(index_dir=index_dir)
        manager.build_job_index(job_embeddings, job_uuids)
        manager.save()

        engine = SemanticSearchEngine(
            db_path=str(db_with_companies.db_path),
            index_dir=index_dir,
        )
        engine.load()

        request = CompanySimilarityRequest(
            company_name="Google Asia Pacific",
            limit=5,
        )

        results = engine.find_similar_companies(request)

        # Should still return results via fallback
        assert len(results) > 0
        result_names = [r.company_name for r in results]
        assert "Google Asia Pacific" not in result_names

    def test_returns_empty_when_fully_degraded(self, temp_dir, db_with_companies):
        """Returns empty list when no indexes are available at all."""
        from src.mcf.embeddings import (
            SemanticSearchEngine,
            CompanySimilarityRequest,
        )

        # No indexes built
        index_dir = temp_dir / "no_indexes"
        index_dir.mkdir()

        engine = SemanticSearchEngine(
            db_path=str(db_with_companies.db_path),
            index_dir=index_dir,
        )
        engine.load()

        results = engine.find_similar_companies(
            CompanySimilarityRequest(company_name="Google Asia Pacific", limit=5)
        )

        assert results == []

    def test_unknown_company_returns_empty(self, temp_dir, db_with_companies):
        """Returns empty list for company with no jobs."""
        from src.mcf.embeddings import (
            FAISSIndexManager,
            SemanticSearchEngine,
            CompanySimilarityRequest,
        )

        index_dir = temp_dir / "indexes"
        index_dir.mkdir()

        job_uuids, job_embeddings = db_with_companies.get_all_embeddings("job")
        manager = FAISSIndexManager(index_dir=index_dir)
        manager.build_job_index(job_embeddings, job_uuids)
        manager.save()

        engine = SemanticSearchEngine(
            db_path=str(db_with_companies.db_path),
            index_dir=index_dir,
        )
        engine.load()

        results = engine.find_similar_companies(
            CompanySimilarityRequest(company_name="Nonexistent Corp", limit=5)
        )

        assert results == []


# =============================================================================
# Integration: generate_all includes company centroids
# =============================================================================


class TestGenerateAllIncludesCompanies:
    """Test that generate_all() now includes company centroid generation."""

    def test_stats_include_companies_processed(self, db_with_companies):
        """generate_all() populates companies_processed in stats."""
        from src.mcf.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator()

        # Mock the model to avoid loading actual sentence-transformers
        mock_embeddings = np.random.randn(384).astype(np.float32)
        mock_embeddings = mock_embeddings / np.linalg.norm(mock_embeddings)

        with patch.object(generator, '_model') as mock_model:
            mock_model.encode = lambda text, **kwargs: (
                mock_embeddings if isinstance(text, str)
                else np.tile(mock_embeddings, (len(text), 1))
            )
            # Directly set the model so the property doesn't re-load
            generator._model = mock_model

            stats = generator.generate_all(
                db_with_companies,
                skip_existing=True,  # Jobs already have embeddings
            )

        assert stats.companies_processed == 3
