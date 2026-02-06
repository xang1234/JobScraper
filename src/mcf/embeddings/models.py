"""
Models for embedding generation and semantic search.

Contains:
- Statistics and configuration models for EmbeddingGenerator
- Request/response models for SemanticSearchEngine
"""

from dataclasses import dataclass, field
from datetime import date, datetime
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


# =============================================================================
# Semantic Search Request/Response Models
# =============================================================================


@dataclass
class SearchRequest:
    """
    Request for semantic search.

    Supports hybrid semantic + keyword search with filters.

    Example:
        request = SearchRequest(
            query="machine learning engineer",
            salary_min=10000,
            employment_type="Full Time",
            limit=20
        )
    """

    query: str
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    employment_type: Optional[str] = None
    company: Optional[str] = None
    region: Optional[str] = None
    limit: int = 20
    min_similarity: float = 0.0
    alpha: float = 0.7  # Weight for semantic vs keyword (1.0 = semantic only)
    expand_query: bool = True
    freshness_weight: float = 0.1  # Boost for recent postings

    def cache_key(self) -> str:
        """Generate a cache key for this request."""
        parts = [
            self.query,
            str(self.salary_min),
            str(self.salary_max),
            str(self.employment_type),
            str(self.company),
            str(self.region),
            str(self.limit),
            str(self.min_similarity),
            str(self.alpha),
            str(self.expand_query),
            str(self.freshness_weight),
        ]
        return "|".join(parts)


@dataclass
class JobResult:
    """
    Single job result from semantic search.

    Contains job details plus relevance score.
    """

    uuid: str
    title: str
    company_name: str
    description: str
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    employment_type: Optional[str] = None
    skills: Optional[str] = None
    location: Optional[str] = None
    posted_date: Optional[date] = None
    job_url: Optional[str] = None
    similarity_score: float = 0.0


@dataclass
class SearchResponse:
    """
    Response from semantic search.

    Contains results plus metadata about the search.
    """

    results: list[JobResult] = field(default_factory=list)
    total_candidates: int = 0
    search_time_ms: float = 0.0
    query_expansion: Optional[list[str]] = None
    degraded: bool = False  # True if fallback to keyword-only was used
    cache_hit: bool = False


@dataclass
class SimilarJobsRequest:
    """
    Request to find jobs similar to a given job.
    """

    job_uuid: str
    limit: int = 10
    exclude_same_company: bool = True
    freshness_weight: float = 0.1


@dataclass
class SkillSearchRequest:
    """
    Request to search jobs by skill similarity.
    """

    skill: str
    limit: int = 20
    min_similarity: float = 0.3
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    employment_type: Optional[str] = None


@dataclass
class CompanySimilarityRequest:
    """
    Request to find companies with similar job profiles.
    """

    company_name: str
    limit: int = 10


@dataclass
class CompanySimilarity:
    """
    Company similarity result.
    """

    company_name: str
    similarity_score: float
    job_count: int = 0
    avg_salary: Optional[int] = None
    top_skills: list[str] = field(default_factory=list)
