"""
Embedding generation and indexing module for semantic search.

Provides vector embeddings for jobs, skills, and companies using
Sentence Transformers (all-MiniLM-L6-v2 model, 384 dimensions),
plus FAISS index management for efficient similarity search.

Features:
- Lazy model loading (defers 2-3s load time until first use)
- Batch processing with progress tracking
- Skill clustering for query expansion
- Company multi-centroid embeddings
- FAISS index management (IVFFlat for jobs, Flat for skills/companies)

Example:
    from src.mcf.embeddings import EmbeddingGenerator, EmbeddingStats, FAISSIndexManager

    generator = EmbeddingGenerator()

    # Single job embedding
    embedding = generator.generate_job_embedding(job)

    # Batch generation with progress
    def on_progress(stats: EmbeddingStats):
        print(f"Progress: {stats.progress_pct:.1f}%")

    stats = generator.generate_all(db, progress_callback=on_progress)

    # Build FAISS index for search
    manager = FAISSIndexManager()
    manager.build_job_index(embeddings, uuids)
    results = manager.search_jobs(query_vector, k=10)
"""

from .models import EmbeddingStats, SkillClusterResult
from .generator import EmbeddingGenerator
from .index_manager import (
    FAISSIndexManager,
    IndexNotBuiltError,
    IndexCompatibilityError,
)

__all__ = [
    "EmbeddingGenerator",
    "EmbeddingStats",
    "SkillClusterResult",
    "FAISSIndexManager",
    "IndexNotBuiltError",
    "IndexCompatibilityError",
]
