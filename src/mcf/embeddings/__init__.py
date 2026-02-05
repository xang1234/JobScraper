"""
Embedding generation module for semantic search.

Provides vector embeddings for jobs, skills, and companies using
Sentence Transformers (all-MiniLM-L6-v2 model, 384 dimensions).

Features:
- Lazy model loading (defers 2-3s load time until first use)
- Batch processing with progress tracking
- Skill clustering for query expansion
- Company multi-centroid embeddings

Example:
    from src.mcf.embeddings import EmbeddingGenerator, EmbeddingStats

    generator = EmbeddingGenerator()

    # Single job embedding
    embedding = generator.generate_job_embedding(job)

    # Batch generation with progress
    def on_progress(stats: EmbeddingStats):
        print(f"Progress: {stats.progress_pct:.1f}%")

    stats = generator.generate_all(db, progress_callback=on_progress)
"""

from .models import EmbeddingStats, SkillClusterResult
from .generator import EmbeddingGenerator

__all__ = [
    "EmbeddingGenerator",
    "EmbeddingStats",
    "SkillClusterResult",
]
