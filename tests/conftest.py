"""
Shared pytest fixtures for all tests.

This module provides reusable fixtures for:
- Temporary directories and files
- Database instances with sample data
- Mock data for various test scenarios

Fixtures are designed to be composable - use `test_db` for database tests,
add `test_embeddings` on top for embedding tests, etc.
"""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from src.mcf.database import MCFDatabase
from src.mcf.models import Job

from .factories import (
    generate_test_jobs,
    generate_similar_jobs,
    generate_company_job_set,
    generate_salary_range_jobs,
)


# =============================================================================
# Basic Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Provide a temporary directory for test files.

    The directory is automatically cleaned up after the test completes.
    Use this for any test that needs to write files.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db_path(temp_dir: Path) -> Path:
    """
    Provide a path for a temporary test database.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path for test database file
    """
    return temp_dir / "test_mcf.db"


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture
def empty_db(temp_db_path: Path) -> MCFDatabase:
    """
    Provide an empty test database.

    Schema is created but no data is inserted.
    Use this for testing database operations from scratch.

    Args:
        temp_db_path: Path for test database

    Returns:
        Empty MCFDatabase instance
    """
    return MCFDatabase(str(temp_db_path))


@pytest.fixture
def test_db(temp_db_path: Path) -> MCFDatabase:
    """
    Provide a test database with sample job data.

    Creates 20 random jobs for general testing purposes.

    Args:
        temp_db_path: Path for test database

    Returns:
        MCFDatabase instance with sample data
    """
    db = MCFDatabase(str(temp_db_path))

    # Insert sample jobs
    jobs = generate_test_jobs(20)
    for job in jobs:
        db.upsert_job(job)

    return db


@pytest.fixture
def large_test_db(temp_db_path: Path) -> MCFDatabase:
    """
    Provide a test database with a larger dataset (100 jobs).

    Use this for performance testing or pagination tests.

    Args:
        temp_db_path: Path for test database

    Returns:
        MCFDatabase instance with 100 jobs
    """
    db = MCFDatabase(str(temp_db_path))

    jobs = generate_test_jobs(100)
    for job in jobs:
        db.upsert_job(job)

    return db


@pytest.fixture
def test_db_with_similar_jobs(temp_db_path: Path) -> tuple[MCFDatabase, list[Job]]:
    """
    Provide a database with groups of similar jobs for similarity testing.

    Creates jobs in clusters:
    - 10 "Data Scientist" variants
    - 10 "Software Engineer" variants
    - 10 "DevOps Engineer" variants

    Args:
        temp_db_path: Path for test database

    Returns:
        Tuple of (database, list of similar job groups)
    """
    db = MCFDatabase(str(temp_db_path))

    all_jobs = []
    for base_title in ["Data Scientist", "Software Engineer", "DevOps Engineer"]:
        jobs = generate_similar_jobs(base_title, n=10)
        for job in jobs:
            db.upsert_job(job)
        all_jobs.extend(jobs)

    return db, all_jobs


@pytest.fixture
def test_db_with_company_groups(temp_db_path: Path) -> MCFDatabase:
    """
    Provide a database with jobs grouped by company.

    Creates 5 jobs each for Google, Meta, and Amazon.
    Use this for testing company statistics and grouping.

    Args:
        temp_db_path: Path for test database

    Returns:
        MCFDatabase instance with company-grouped jobs
    """
    db = MCFDatabase(str(temp_db_path))

    for company in ["Google Asia Pacific", "Meta Platforms Singapore", "Amazon Web Services"]:
        jobs = generate_company_job_set(company, n=5)
        for job in jobs:
            db.upsert_job(job)

    return db


# =============================================================================
# Embedding Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding() -> np.ndarray:
    """
    Generate a single mock embedding vector.

    Returns a normalized 384-dimensional vector (MiniLM embedding size).

    Returns:
        Normalized numpy array of shape (384,)
    """
    embedding = np.random.randn(384).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize
    return embedding


@pytest.fixture
def mock_embeddings_batch() -> tuple[list[str], np.ndarray]:
    """
    Generate a batch of mock embeddings with UUIDs.

    Returns 20 random embeddings suitable for batch operations.

    Returns:
        Tuple of (uuids, embeddings_matrix)
        embeddings_matrix has shape (20, 384)
    """
    import uuid

    n = 20
    uuids = [str(uuid.uuid4()) for _ in range(n)]

    embeddings = np.random.randn(n, 384).astype(np.float32)
    # Normalize each row
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return uuids, embeddings


@pytest.fixture
def test_db_with_embeddings(test_db: MCFDatabase) -> MCFDatabase:
    """
    Provide a database with jobs and their embeddings.

    Generates random embeddings for all jobs in the test database.
    Use this for testing embedding retrieval and similarity calculations.

    Args:
        test_db: Database fixture with sample jobs

    Returns:
        MCFDatabase with embeddings for all jobs
    """
    # Get all job UUIDs
    jobs = test_db.search_jobs(limit=1000)

    for job in jobs:
        # Generate random embedding
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        test_db.upsert_embedding(
            entity_id=job["uuid"],
            entity_type="job",
            embedding=embedding,
        )

    return test_db


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_job() -> Job:
    """
    Provide a single sample job for basic tests.

    Returns:
        A single Job object
    """
    from .factories import generate_test_job

    return generate_test_job(
        title="Senior Data Scientist",
        company_name="Test Company Pte Ltd",
        salary_min=10000,
        salary_max=15000,
    )


@pytest.fixture
def sample_jobs() -> list[Job]:
    """
    Provide a small batch of sample jobs.

    Returns:
        List of 5 Job objects
    """
    return generate_test_jobs(5)


@pytest.fixture
def high_salary_jobs() -> list[Job]:
    """
    Provide jobs with high salaries (10k-20k range).

    Use for testing salary filters.

    Returns:
        List of 10 high-salary Job objects
    """
    return generate_salary_range_jobs(min_salary=10000, max_salary=20000, n=10)


@pytest.fixture
def low_salary_jobs() -> list[Job]:
    """
    Provide jobs with lower salaries (3k-6k range).

    Use for testing salary filters.

    Returns:
        List of 10 low-salary Job objects
    """
    return generate_salary_range_jobs(min_salary=3000, max_salary=6000, n=10)


# =============================================================================
# Skip Markers for Optional Features
# =============================================================================


def requires_embeddings(func):
    """
    Decorator to skip tests if embedding infrastructure is not available.

    Use this for tests that depend on EmbeddingGenerator or FAISSIndexManager.
    """
    return pytest.mark.embedding(func)


def requires_faiss(func):
    """
    Decorator to skip tests if FAISS is not available.
    """
    try:
        import faiss  # noqa: F401

        return func
    except ImportError:
        return pytest.mark.skip(reason="FAISS not installed")(func)


def requires_sentence_transformers(func):
    """
    Decorator to skip tests if sentence-transformers is not available.
    """
    try:
        import sentence_transformers  # noqa: F401

        return func
    except ImportError:
        return pytest.mark.skip(reason="sentence-transformers not installed")(func)
