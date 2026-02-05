"""
Pydantic models for embedding generation.

Contains statistics and configuration models for the EmbeddingGenerator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EmbeddingStats:
    """
    Statistics from embedding generation.

    Tracks progress and performance metrics during batch embedding generation.
    Used for progress reporting and post-run analysis.

    Example:
        stats = EmbeddingStats(jobs_total=10000)
        for batch in batches:
            stats.jobs_processed += len(batch)
            print(f"Progress: {stats.jobs_processed}/{stats.jobs_total}")
    """

    jobs_total: int = 0
    jobs_processed: int = 0
    jobs_skipped: int = 0
    jobs_failed: int = 0
    unique_skills: int = 0
    skill_clusters: int = 0
    companies_processed: int = 0
    elapsed_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def jobs_per_second(self) -> float:
        """Calculate processing throughput."""
        if self.elapsed_seconds > 0:
            return self.jobs_processed / self.elapsed_seconds
        return 0.0

    @property
    def progress_pct(self) -> float:
        """Calculate completion percentage."""
        if self.jobs_total > 0:
            return (self.jobs_processed / self.jobs_total) * 100
        return 0.0

    @property
    def is_complete(self) -> bool:
        """Check if all jobs have been processed."""
        return self.jobs_processed >= self.jobs_total and self.jobs_total > 0


@dataclass
class SkillClusterResult:
    """
    Result from skill clustering operation.

    Provides cluster assignments and centroids for query expansion.

    Attributes:
        clusters: Mapping of cluster_id -> list of skill names
        skill_to_cluster: Mapping of skill_name -> cluster_id
        cluster_centroids: Mapping of cluster_id -> centroid embedding (as list)
    """

    clusters: dict[int, list[str]] = field(default_factory=dict)
    skill_to_cluster: dict[str, int] = field(default_factory=dict)
    cluster_centroids: dict[int, list[float]] = field(default_factory=dict)

    @property
    def num_clusters(self) -> int:
        """Number of skill clusters."""
        return len(self.clusters)

    @property
    def num_skills(self) -> int:
        """Total number of skills clustered."""
        return len(self.skill_to_cluster)

    def get_related_skills(self, skill: str) -> list[str]:
        """
        Get skills in the same cluster as the given skill.

        Args:
            skill: Skill name to find related skills for

        Returns:
            List of related skill names (including the input skill)
        """
        cluster_id = self.skill_to_cluster.get(skill)
        if cluster_id is not None:
            return self.clusters.get(cluster_id, [])
        return []
