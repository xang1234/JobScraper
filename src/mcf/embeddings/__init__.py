"""
Embedding generation, indexing, and semantic search module.

Provides:
- Vector embeddings for jobs, skills, and companies (Sentence Transformers)
- FAISS index management for efficient similarity search
- SemanticSearchEngine for hybrid semantic + keyword search

Features:
- Lazy model loading (defers 2-3s load time until first use)
- Batch processing with progress tracking
- Skill clustering for query expansion
- Company multi-centroid embeddings
- FAISS index management (IVFFlat for jobs, Flat for skills/companies)
- Hybrid scoring combining semantic and BM25 keyword search
- Result caching and graceful degradation

Example:
    from src.mcf.embeddings import SemanticSearchEngine, SearchRequest

    # Create and load the search engine
    engine = SemanticSearchEngine("data/mcf_jobs.db")
    engine.load()

    # Perform semantic search
    response = engine.search(SearchRequest(
        query="machine learning engineer",
        salary_min=10000,
        limit=20
    ))

    for job in response.results:
        print(f"{job.title} at {job.company_name}: {job.similarity_score:.3f}")
"""

from .models import (
    EmbeddingStats,
    SkillClusterResult,
    SearchRequest,
    SearchResponse,
    JobResult,
    SimilarJobsRequest,
    SkillSearchRequest,
    CompanySimilarityRequest,
    CompanySimilarity,
)
from .generator import EmbeddingGenerator
from .index_manager import (
    FAISSIndexManager,
    IndexNotBuiltError,
    IndexCompatibilityError,
)
from .query_expander import QueryExpander
from .search_engine import SemanticSearchEngine

__all__ = [
    # Generator
    "EmbeddingGenerator",
    "EmbeddingStats",
    "SkillClusterResult",
    # Index management
    "FAISSIndexManager",
    "IndexNotBuiltError",
    "IndexCompatibilityError",
    # Query expansion
    "QueryExpander",
    # Search engine
    "SemanticSearchEngine",
    "SearchRequest",
    "SearchResponse",
    "JobResult",
    "SimilarJobsRequest",
    "SkillSearchRequest",
    "CompanySimilarityRequest",
    "CompanySimilarity",
]
