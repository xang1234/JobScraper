"""
Tests for FAISSIndexManager.

Tests cover:
- Index building (jobs, skills, companies)
- Search operations (basic, filtered, skills, companies)
- Persistence (save/load)
- Edge cases and error handling
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_faiss


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_index_dir():
    """Provide a temporary directory for index files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    n = 1000
    embeddings = np.random.randn(n, 384).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    uuids = [f"uuid-{i}" for i in range(n)]
    return embeddings, uuids


@pytest.fixture
def small_embeddings():
    """Generate small set of embeddings for basic tests."""
    np.random.seed(123)
    n = 50
    embeddings = np.random.randn(n, 384).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    uuids = [f"small-uuid-{i}" for i in range(n)]
    return embeddings, uuids


@pytest.fixture
def skill_embeddings():
    """Generate skill embeddings for testing."""
    np.random.seed(456)
    skills = ["Python", "Java", "JavaScript", "SQL", "Machine Learning",
              "Data Analysis", "AWS", "Docker", "Kubernetes", "React"]
    n = len(skills)
    embeddings = np.random.randn(n, 384).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings, skills


@pytest.fixture
def company_centroids():
    """Generate company centroids for testing."""
    np.random.seed(789)

    def make_centroids(n):
        centroids = np.random.randn(n, 384).astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        return [c for c in centroids / norms]

    return {
        "Google": make_centroids(3),  # Large company, multiple centroids
        "Meta": make_centroids(2),
        "Startup Inc": make_centroids(1),  # Small company, single centroid
    }


# =============================================================================
# Build Tests
# =============================================================================


@requires_faiss
class TestBuildJobIndex:
    """Tests for build_job_index method."""

    def test_build_job_index_basic(self, temp_index_dir, sample_embeddings):
        """Test basic job index building."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)

        manager.build_job_index(embeddings, uuids)

        assert "jobs" in manager.indexes
        assert manager.indexes["jobs"].ntotal == len(uuids)
        assert len(manager.uuid_to_idx) == len(uuids)
        assert len(manager.idx_to_uuid) == len(uuids)

    def test_build_job_index_small_dataset(self, temp_index_dir, small_embeddings):
        """Test that small datasets use Flat index instead of IVFFlat."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = small_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)

        manager.build_job_index(embeddings, uuids)

        # Small dataset should use IndexFlatIP
        assert "Flat" in type(manager.indexes["jobs"]).__name__

    def test_build_job_index_mismatched_lengths(self, temp_index_dir):
        """Test error when embeddings and uuids have different lengths."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings = np.random.randn(100, 384).astype(np.float32)
        uuids = [f"uuid-{i}" for i in range(50)]  # Mismatch

        manager = FAISSIndexManager(index_dir=temp_index_dir)

        with pytest.raises(ValueError, match="must have same length"):
            manager.build_job_index(embeddings, uuids)

    def test_build_job_index_custom_nlist(self, temp_index_dir, sample_embeddings):
        """Test building with custom nlist."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)

        manager.build_job_index(embeddings, uuids, nlist=64)

        assert manager._nlist == 64


@requires_faiss
class TestBuildSkillIndex:
    """Tests for build_skill_index method."""

    def test_build_skill_index_basic(self, temp_index_dir, skill_embeddings):
        """Test basic skill index building."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, skills = skill_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)

        manager.build_skill_index(embeddings, skills)

        assert "skills" in manager.indexes
        assert manager.indexes["skills"].ntotal == len(skills)
        assert manager.skill_names == skills

    def test_build_skill_index_mismatched_lengths(self, temp_index_dir):
        """Test error when embeddings and skill_names have different lengths."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings = np.random.randn(10, 384).astype(np.float32)
        skills = ["Skill1", "Skill2"]  # Mismatch

        manager = FAISSIndexManager(index_dir=temp_index_dir)

        with pytest.raises(ValueError, match="must have same length"):
            manager.build_skill_index(embeddings, skills)


@requires_faiss
class TestBuildCompanyIndex:
    """Tests for build_company_index method."""

    def test_build_company_index_basic(self, temp_index_dir, company_centroids):
        """Test basic company index building."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_index_dir)

        manager.build_company_index(company_centroids)

        assert "companies" in manager.indexes
        # Total centroids = 3 + 2 + 1 = 6
        assert manager.indexes["companies"].ntotal == 6
        assert len(manager.company_names) == 3

    def test_build_company_index_empty(self, temp_index_dir):
        """Test building with empty company centroids."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_index_dir)

        manager.build_company_index({})

        assert "companies" not in manager.indexes


# =============================================================================
# Search Tests
# =============================================================================


@requires_faiss
class TestSearchJobs:
    """Tests for search_jobs method."""

    def test_search_jobs_basic(self, temp_index_dir, sample_embeddings):
        """Test basic job search."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_job_index(embeddings, uuids)

        # Search with first embedding - should find itself
        results = manager.search_jobs(embeddings[0], k=5)

        assert len(results) == 5
        assert results[0][0] == "uuid-0"  # First result is query itself
        assert results[0][1] > 0.99  # Very high similarity (nearly 1.0)

    def test_search_jobs_returns_k_results(self, temp_index_dir, sample_embeddings):
        """Test that search returns exactly k results."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_job_index(embeddings, uuids)

        for k in [1, 5, 10, 50]:
            results = manager.search_jobs(embeddings[0], k=k)
            assert len(results) == k

    def test_search_jobs_not_built(self, temp_index_dir):
        """Test error when searching unbuilt index."""
        from src.mcf.embeddings import FAISSIndexManager, IndexNotBuiltError

        manager = FAISSIndexManager(index_dir=temp_index_dir)
        query = np.random.randn(384).astype(np.float32)

        with pytest.raises(IndexNotBuiltError):
            manager.search_jobs(query)

    def test_search_jobs_custom_nprobe(self, temp_index_dir, sample_embeddings):
        """Test search with custom nprobe."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_job_index(embeddings, uuids)

        # Higher nprobe should still work
        results = manager.search_jobs(embeddings[0], k=5, nprobe=50)

        assert len(results) == 5


@requires_faiss
class TestSearchJobsFiltered:
    """Tests for search_jobs_filtered method."""

    def test_filtered_search_basic(self, temp_index_dir, sample_embeddings):
        """Test filtered search with subset of UUIDs."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_job_index(embeddings, uuids)

        # Only allow UUIDs 0-99
        allowed = {f"uuid-{i}" for i in range(100)}
        results = manager.search_jobs_filtered(embeddings[0], allowed, k=5)

        assert len(results) == 5
        # All results should be in allowed set
        for uuid, _ in results:
            assert uuid in allowed

    def test_filtered_search_empty_allowed(self, temp_index_dir, sample_embeddings):
        """Test filtered search with empty allowed set."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_job_index(embeddings, uuids)

        results = manager.search_jobs_filtered(embeddings[0], set(), k=5)

        assert results == []

    def test_filtered_search_single_allowed(self, temp_index_dir, sample_embeddings):
        """Test filtered search with single UUID allowed."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_job_index(embeddings, uuids)

        allowed = {"uuid-50"}
        results = manager.search_jobs_filtered(embeddings[0], allowed, k=5)

        assert len(results) == 1
        assert results[0][0] == "uuid-50"


@requires_faiss
class TestSearchSkills:
    """Tests for search_skills method."""

    def test_search_skills_basic(self, temp_index_dir, skill_embeddings):
        """Test basic skill search."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, skills = skill_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_skill_index(embeddings, skills)

        # Search with Python embedding
        results = manager.search_skills(embeddings[0], k=3)

        assert len(results) == 3
        assert results[0][0] == "Python"  # First result is query itself

    def test_search_skills_not_built(self, temp_index_dir):
        """Test error when searching unbuilt skill index."""
        from src.mcf.embeddings import FAISSIndexManager, IndexNotBuiltError

        manager = FAISSIndexManager(index_dir=temp_index_dir)
        query = np.random.randn(384).astype(np.float32)

        with pytest.raises(IndexNotBuiltError):
            manager.search_skills(query)


@requires_faiss
class TestSearchCompanies:
    """Tests for search_companies method."""

    def test_search_companies_basic(self, temp_index_dir, company_centroids):
        """Test basic company search."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_company_index(company_centroids)

        # Search with Google's first centroid
        query = company_centroids["Google"][0]
        results = manager.search_companies(query, k=3)

        assert len(results) == 3
        assert results[0][0] == "Google"  # Google should be first

    def test_search_companies_not_built(self, temp_index_dir):
        """Test error when searching unbuilt company index."""
        from src.mcf.embeddings import FAISSIndexManager, IndexNotBuiltError

        manager = FAISSIndexManager(index_dir=temp_index_dir)
        query = np.random.randn(384).astype(np.float32)

        with pytest.raises(IndexNotBuiltError):
            manager.search_companies(query)


# =============================================================================
# Persistence Tests
# =============================================================================


@requires_faiss
class TestPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load_jobs(self, temp_index_dir, sample_embeddings):
        """Test saving and loading job index."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings

        # Build and save
        manager1 = FAISSIndexManager(index_dir=temp_index_dir)
        manager1.build_job_index(embeddings, uuids)
        manager1.save()

        # Load in new instance
        manager2 = FAISSIndexManager(index_dir=temp_index_dir)
        assert manager2.load()

        # Verify same results
        results1 = manager1.search_jobs(embeddings[0], k=5)
        results2 = manager2.search_jobs(embeddings[0], k=5)

        assert results1 == results2

    def test_save_and_load_all_indexes(
        self, temp_index_dir, sample_embeddings, skill_embeddings, company_centroids
    ):
        """Test saving and loading all index types."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        skill_embs, skills = skill_embeddings

        # Build and save
        manager1 = FAISSIndexManager(index_dir=temp_index_dir)
        manager1.build_job_index(embeddings, uuids)
        manager1.build_skill_index(skill_embs, skills)
        manager1.build_company_index(company_centroids)
        manager1.save()

        # Load in new instance
        manager2 = FAISSIndexManager(index_dir=temp_index_dir)
        assert manager2.load()

        # Verify all indexes loaded
        assert "jobs" in manager2.indexes
        assert "skills" in manager2.indexes
        assert "companies" in manager2.indexes

    def test_load_nonexistent(self, temp_index_dir):
        """Test loading from empty directory."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_index_dir)

        assert not manager.load()

    def test_exists_check(self, temp_index_dir, sample_embeddings):
        """Test exists() method."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings

        manager = FAISSIndexManager(index_dir=temp_index_dir)

        assert not manager.exists()

        manager.build_job_index(embeddings, uuids)
        manager.save()

        assert manager.exists()

    def test_model_version_compatibility(self, temp_index_dir, sample_embeddings):
        """Test that incompatible model versions raise error."""
        from src.mcf.embeddings import FAISSIndexManager, IndexCompatibilityError

        embeddings, uuids = sample_embeddings

        # Build with one version
        manager1 = FAISSIndexManager(
            index_dir=temp_index_dir,
            model_version="all-MiniLM-L6-v2",
        )
        manager1.build_job_index(embeddings, uuids)
        manager1.save()

        # Try to load with different version
        manager2 = FAISSIndexManager(
            index_dir=temp_index_dir,
            model_version="different-model",
        )

        with pytest.raises(IndexCompatibilityError):
            manager2.load()


# =============================================================================
# Update Tests
# =============================================================================


@requires_faiss
class TestUpdateMethods:
    """Tests for add_jobs and remove_jobs methods."""

    def test_add_jobs(self, temp_index_dir, sample_embeddings):
        """Test adding jobs to existing index."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_job_index(embeddings, uuids)

        initial_count = manager.indexes["jobs"].ntotal

        # Add new jobs
        new_embeddings = np.random.randn(10, 384).astype(np.float32)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        new_uuids = [f"new-uuid-{i}" for i in range(10)]

        manager.add_jobs(new_embeddings, new_uuids)

        assert manager.indexes["jobs"].ntotal == initial_count + 10
        assert "new-uuid-0" in manager.uuid_to_idx

    def test_remove_jobs(self, temp_index_dir, sample_embeddings):
        """Test removing jobs from index (soft delete)."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_job_index(embeddings, uuids)

        # Remove some jobs
        to_remove = ["uuid-0", "uuid-1", "uuid-2"]
        manager.remove_jobs(to_remove)

        # UUIDs should be gone from mapping
        assert "uuid-0" not in manager.uuid_to_idx
        assert "uuid-1" not in manager.uuid_to_idx

        # But index size unchanged (soft delete)
        assert manager.indexes["jobs"].ntotal == len(uuids)


# =============================================================================
# Utility Tests
# =============================================================================


@requires_faiss
class TestUtilities:
    """Tests for utility methods."""

    def test_get_stats(self, temp_index_dir, sample_embeddings, skill_embeddings):
        """Test get_stats method."""
        from src.mcf.embeddings import FAISSIndexManager

        embeddings, uuids = sample_embeddings
        skill_embs, skills = skill_embeddings

        manager = FAISSIndexManager(index_dir=temp_index_dir)
        manager.build_job_index(embeddings, uuids)
        manager.build_skill_index(skill_embs, skills)

        stats = manager.get_stats()

        assert "indexes" in stats
        assert "jobs" in stats["indexes"]
        assert "skills" in stats["indexes"]
        assert stats["indexes"]["jobs"]["total_vectors"] == len(uuids)
        assert stats["indexes"]["skills"]["total_skills"] == len(skills)

    def test_is_compatible(self, temp_index_dir):
        """Test is_compatible method."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(
            index_dir=temp_index_dir,
            model_version="all-MiniLM-L6-v2",
        )

        assert manager.is_compatible("all-MiniLM-L6-v2")
        assert not manager.is_compatible("different-model")

    def test_calculate_nlist(self, temp_index_dir):
        """Test nlist calculation."""
        from src.mcf.embeddings import FAISSIndexManager

        manager = FAISSIndexManager(index_dir=temp_index_dir)

        # Small dataset: minimum nlist
        assert manager._calculate_nlist(100) >= 16

        # Medium dataset: sqrt(n)
        nlist_10k = manager._calculate_nlist(10000)
        assert 90 <= nlist_10k <= 110  # Around sqrt(10000) = 100

        # Large dataset: capped
        assert manager._calculate_nlist(100_000_000) <= 4096
