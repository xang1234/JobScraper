"""
Semantic Search Engine for MCF job data.

Orchestrates hybrid semantic + keyword search by combining:
- SQL filtering (salary, location, employment type)
- FAISS vector search (semantic similarity)
- FTS5 BM25 search (keyword relevance)
- Query expansion (synonym matching via skill clusters)
- Result caching (performance optimization)
- Graceful degradation (reliability when indexes unavailable)

Example:
    engine = SemanticSearchEngine(db_path="data/mcf_jobs.db")
    engine.load()

    response = engine.search(SearchRequest(
        query="machine learning engineer",
        salary_min=10000,
        limit=20
    ))

    for job in response.results:
        print(f"{job.title} at {job.company_name}: {job.similarity_score:.3f}")
"""

import hashlib
import logging
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from cachetools import TTLCache

from ..database import MCFDatabase
from .generator import EmbeddingGenerator
from .index_manager import (
    FAISSIndexManager,
    IndexNotBuiltError,
    IndexCompatibilityError,
)
from .models import (
    CompanySimilarity,
    CompanySimilarityRequest,
    JobResult,
    SearchRequest,
    SearchResponse,
    SimilarJobsRequest,
    SkillSearchRequest,
)
from .query_expander import QueryExpander

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Orchestrates hybrid semantic + keyword search.

    Combines multiple search strategies:
    - SQL filtering for hard constraints (salary, location, type)
    - FAISS vector search for semantic similarity
    - FTS5 BM25 search for keyword relevance
    - Query expansion for synonym matching
    - Result caching for performance
    - Graceful degradation for reliability

    The hybrid scoring formula is:
        score = alpha * semantic_score + (1 - alpha) * bm25_score + freshness_boost

    Example:
        engine = SemanticSearchEngine("data/mcf_jobs.db")
        engine.load()

        response = engine.search(SearchRequest(
            query="machine learning engineer",
            salary_min=10000,
            limit=20
        ))
    """

    # Default cache configuration
    QUERY_CACHE_SIZE = 1000
    QUERY_CACHE_TTL = 3600  # 1 hour
    RESULT_CACHE_SIZE = 200
    RESULT_CACHE_TTL = 300  # 5 minutes

    def __init__(
        self,
        db_path: str = "data/mcf_jobs.db",
        index_dir: Path = Path("data/embeddings"),
        model_version: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the search engine.

        Args:
            db_path: Path to SQLite database
            index_dir: Directory containing FAISS indexes
            model_version: Embedding model version for compatibility
        """
        self.db = MCFDatabase(db_path)
        self.index_dir = Path(index_dir)
        self.model_version = model_version

        # Components (loaded lazily)
        self.index_manager = FAISSIndexManager(
            index_dir=self.index_dir,
            model_version=model_version,
        )
        self.generator = EmbeddingGenerator(model_name=model_version)
        self.query_expander: Optional[QueryExpander] = None

        # Caches
        self._query_cache: TTLCache = TTLCache(
            maxsize=self.QUERY_CACHE_SIZE,
            ttl=self.QUERY_CACHE_TTL,
        )
        self._result_cache: TTLCache = TTLCache(
            maxsize=self.RESULT_CACHE_SIZE,
            ttl=self.RESULT_CACHE_TTL,
        )

        # State
        self._loaded = False
        self._degraded = False
        self._has_vector_index = False
        self._has_skill_clusters = False

    def load(self) -> bool:
        """
        Load indexes and prepare for searching.

        Returns:
            True if indexes loaded successfully, False if degraded mode

        Note:
            This method is idempotent - safe to call multiple times.
        """
        if self._loaded:
            return not self._degraded

        logger.info("Loading semantic search engine...")

        # Try to load FAISS indexes
        try:
            if self.index_manager.exists():
                self.index_manager.load()
                self._has_vector_index = True
                logger.info("FAISS indexes loaded successfully")
            else:
                logger.warning(
                    f"FAISS indexes not found at {self.index_dir}. "
                    "Run 'mcf embed-generate' to build indexes."
                )
                self._degraded = True
        except IndexCompatibilityError as e:
            logger.warning(f"Index compatibility error: {e}. Falling back to keyword search.")
            self._degraded = True
        except Exception as e:
            logger.warning(f"Failed to load FAISS indexes: {e}. Falling back to keyword search.")
            self._degraded = True

        # Try to load query expander (skill clusters)
        try:
            self.query_expander = QueryExpander.load(self.index_dir)
            self._has_skill_clusters = True
            logger.info(f"Query expander loaded: {self.query_expander.get_stats()}")
        except FileNotFoundError:
            logger.warning(
                "Skill clusters not found. Query expansion disabled. "
                "Run 'mcf embed-generate' to create clusters."
            )
        except Exception as e:
            logger.warning(f"Failed to load query expander: {e}")

        self._loaded = True
        return not self._degraded

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Main semantic search with all features.

        Flow:
        1. Check cache for identical request
        2. Apply SQL filters to get candidates
        3. Expand query if enabled
        4. Get query embedding (cached)
        5. Compute hybrid scores
        6. Return top k results

        Args:
            request: Search parameters

        Returns:
            SearchResponse with ranked results and metadata
        """
        start_time = time.time()

        # Ensure engine is loaded
        if not self._loaded:
            self.load()

        # Check result cache
        cache_key = request.cache_key()
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            cached.cache_hit = True
            return cached

        try:
            # Step 1: SQL filtering
            candidates = self._apply_sql_filters(request)
            total_candidates = len(candidates)

            if not candidates:
                response = SearchResponse(
                    results=[],
                    total_candidates=0,
                    search_time_ms=(time.time() - start_time) * 1000,
                    degraded=self._degraded,
                )
                return response

            # Step 2: Query expansion
            query_expansion = None
            search_query = request.query

            if request.expand_query and self.query_expander:
                expanded = self.query_expander.expand(request.query)
                if len(expanded) > 1:
                    query_expansion = expanded
                    # Use expanded terms for BM25 search
                    search_query = " ".join(expanded)
                    logger.debug(f"Query expanded: '{request.query}' -> {expanded}")

            # Step 3: Compute hybrid scores
            if self._has_vector_index and not self._degraded:
                scored_results = self._compute_hybrid_scores(
                    query=request.query,
                    search_query=search_query,
                    candidate_uuids=[c["uuid"] for c in candidates],
                    alpha=request.alpha,
                    freshness_weight=request.freshness_weight,
                )
            else:
                # Degraded mode: keyword-only search
                scored_results = self._keyword_only_scores(
                    search_query=search_query,
                    candidate_uuids=[c["uuid"] for c in candidates],
                    freshness_weight=request.freshness_weight,
                )

            # Step 4: Filter by minimum similarity and limit
            filtered_results = [
                (uuid, score)
                for uuid, score in scored_results
                if score >= request.min_similarity
            ][: request.limit]

            # Step 5: Enrich with full job data
            results = self._enrich_results(filtered_results)

            response = SearchResponse(
                results=results,
                total_candidates=total_candidates,
                search_time_ms=(time.time() - start_time) * 1000,
                query_expansion=query_expansion,
                degraded=self._degraded,
                cache_hit=False,
            )

            # Cache result
            self._result_cache[cache_key] = response

            # Log analytics
            self._log_search(request, response)

            return response

        except (IndexNotBuiltError, IndexCompatibilityError) as e:
            logger.warning(f"Vector search failed: {e}. Falling back to keyword search.")
            self._degraded = True
            return self._keyword_fallback_search(request, start_time)

    def find_similar(self, request: SimilarJobsRequest) -> SearchResponse:
        """
        Find jobs similar to a given job.

        Uses the job's own embedding as the query vector.

        Args:
            request: Similar jobs request parameters

        Returns:
            SearchResponse with similar jobs
        """
        start_time = time.time()

        if not self._loaded:
            self.load()

        if self._degraded or not self._has_vector_index:
            return SearchResponse(
                results=[],
                total_candidates=0,
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=True,
            )

        # Get source job embedding
        source_embedding = self.db.get_embedding(request.job_uuid, "job")
        if source_embedding is None:
            logger.warning(f"No embedding found for job {request.job_uuid}")
            return SearchResponse(
                results=[],
                total_candidates=0,
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=self._degraded,
            )

        # Get source job for company exclusion
        source_job = self.db.get_job(request.job_uuid)

        # Search for similar (request extra to allow for filtering)
        search_k = request.limit + 10 if request.exclude_same_company else request.limit + 1

        try:
            results = self.index_manager.search_jobs(source_embedding, k=search_k)
        except IndexNotBuiltError:
            return SearchResponse(
                results=[],
                total_candidates=0,
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=True,
            )

        # Filter results
        filtered: list[tuple[str, float]] = []
        for uuid, score in results:
            # Skip the source job itself
            if uuid == request.job_uuid:
                continue

            # Skip same company if requested
            if request.exclude_same_company and source_job:
                job = self.db.get_job(uuid)
                if job and job.get("company_name") == source_job.get("company_name"):
                    continue

            filtered.append((uuid, score))

            if len(filtered) >= request.limit:
                break

        # Apply freshness boost
        if request.freshness_weight > 0 and filtered:
            freshness = self._compute_freshness_scores([uuid for uuid, _ in filtered])
            filtered = [
                (uuid, score + request.freshness_weight * freshness.get(uuid, 0.5))
                for uuid, score in filtered
            ]
            filtered.sort(key=lambda x: x[1], reverse=True)

        results_enriched = self._enrich_results(filtered)

        return SearchResponse(
            results=results_enriched,
            total_candidates=len(results_enriched),
            search_time_ms=(time.time() - start_time) * 1000,
            degraded=self._degraded,
        )

    def search_by_skill(self, request: SkillSearchRequest) -> SearchResponse:
        """
        Search jobs by skill similarity.

        Finds jobs that require skills similar to the specified skill.

        Args:
            request: Skill search parameters

        Returns:
            SearchResponse with matching jobs
        """
        start_time = time.time()

        if not self._loaded:
            self.load()

        # Generate embedding for the skill query
        skill_embedding = self._get_query_embedding(request.skill)

        if self._degraded or not self._has_vector_index:
            # Fall back to keyword search
            return self._keyword_fallback_search(
                SearchRequest(
                    query=request.skill,
                    limit=request.limit,
                    min_similarity=request.min_similarity,
                ),
                start_time,
            )

        try:
            # Search job index with skill embedding
            results = self.index_manager.search_jobs(skill_embedding, k=request.limit * 2)

            # Filter by minimum similarity
            filtered = [
                (uuid, score)
                for uuid, score in results
                if score >= request.min_similarity
            ][: request.limit]

            results_enriched = self._enrich_results(filtered)

            return SearchResponse(
                results=results_enriched,
                total_candidates=len(filtered),
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=self._degraded,
            )

        except IndexNotBuiltError:
            return SearchResponse(
                results=[],
                total_candidates=0,
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=True,
            )

    def find_similar_companies(
        self, request: CompanySimilarityRequest
    ) -> list[CompanySimilarity]:
        """
        Find companies with similar job profiles.

        Primary: Uses pre-computed multi-centroid company index for efficient matching.
        For each source centroid, searches the company index and aggregates by
        max similarity across all centroid pairs between source and target companies.

        Fallback: If company index is unavailable, computes a single centroid on-the-fly
        from the source company's job embeddings and searches the jobs index.

        Args:
            request: Company similarity request

        Returns:
            List of similar companies with stats
        """
        if not self._loaded:
            self.load()

        if self._degraded or not self._has_vector_index:
            return []

        # Check source company exists
        source_stats = self.db.get_company_stats(request.company_name)
        if source_stats.get("job_count", 0) == 0:
            logger.warning(f"No jobs found for company: {request.company_name}")
            return []

        # Primary path: use pre-computed company multi-centroid index
        if self.index_manager.has_company_index():
            source_centroids = self.index_manager.get_company_centroids(
                request.company_name
            )
            if source_centroids is not None:
                return self._find_similar_companies_multi_centroid(
                    request, source_centroids
                )

        # Fallback: on-the-fly single centroid via jobs index
        return self._find_similar_companies_fallback(request)

    def _find_similar_companies_multi_centroid(
        self,
        request: CompanySimilarityRequest,
        source_centroids: np.ndarray,
    ) -> list[CompanySimilarity]:
        """
        Find similar companies using multi-centroid matching.

        For each source centroid, searches the company index. For each target
        company, takes the max similarity across all source-target centroid pairs.

        Args:
            request: Company similarity request
            source_centroids: Source company's centroid array (n_centroids, dim)

        Returns:
            List of CompanySimilarity results
        """
        company_scores: dict[str, float] = {}

        # For each source centroid, search company index
        for centroid in source_centroids:
            try:
                results = self.index_manager.search_companies(
                    centroid, k=request.limit + 10
                )
            except IndexNotBuiltError:
                continue

            for company_name, score in results:
                if company_name == request.company_name:
                    continue
                # Max similarity across all centroid pairs
                if company_name not in company_scores or score > company_scores[company_name]:
                    company_scores[company_name] = score

        # Sort by score descending
        sorted_companies = sorted(
            company_scores.items(), key=lambda x: x[1], reverse=True
        )[: request.limit]

        # Enrich with company stats
        results_list: list[CompanySimilarity] = []
        for company_name, score in sorted_companies:
            stats = self.db.get_company_stats(company_name)
            results_list.append(
                CompanySimilarity(
                    company_name=company_name,
                    similarity_score=score,
                    job_count=stats.get("job_count", 0),
                    avg_salary=stats.get("avg_salary"),
                    top_skills=stats.get("top_skills", [])[:5],
                )
            )

        return results_list

    def _find_similar_companies_fallback(
        self, request: CompanySimilarityRequest
    ) -> list[CompanySimilarity]:
        """
        Fallback: compute single centroid on-the-fly and search jobs index.

        Used when company index is not available (degraded mode).

        Args:
            request: Company similarity request

        Returns:
            List of CompanySimilarity results
        """
        company_jobs = self.db.search_jobs(
            company_name=request.company_name,
            limit=50,
        )

        if not company_jobs:
            return []

        job_uuids = [j["uuid"] for j in company_jobs]
        embeddings_dict = self.db.get_embeddings_for_uuids(job_uuids)

        if not embeddings_dict:
            return []

        # Compute single centroid
        embeddings = np.array(list(embeddings_dict.values()))
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        try:
            similar_jobs = self.index_manager.search_jobs(centroid, k=500)
        except IndexNotBuiltError:
            return []

        # Aggregate by company
        company_scores: dict[str, list[float]] = {}
        for uuid, score in similar_jobs:
            job = self.db.get_job(uuid)
            if job and job.get("company_name"):
                company = job["company_name"]
                if company == request.company_name:
                    continue
                if company not in company_scores:
                    company_scores[company] = []
                company_scores[company].append(score)

        # Take average score per company
        company_avg: list[tuple[str, float]] = [
            (company, sum(scores) / len(scores))
            for company, scores in company_scores.items()
        ]
        company_avg.sort(key=lambda x: x[1], reverse=True)

        # Enrich with company stats
        results: list[CompanySimilarity] = []
        for company_name, score in company_avg[: request.limit]:
            stats = self.db.get_company_stats(company_name)
            results.append(
                CompanySimilarity(
                    company_name=company_name,
                    similarity_score=score,
                    job_count=stats.get("job_count", 0),
                    avg_salary=stats.get("avg_salary"),
                    top_skills=stats.get("top_skills", [])[:5],
                )
            )

        return results

    def get_stats(self) -> dict:
        """
        Get search engine statistics.

        Returns:
            Dict with index stats, cache stats, and engine state
        """
        stats = {
            "loaded": self._loaded,
            "degraded": self._degraded,
            "has_vector_index": self._has_vector_index,
            "has_skill_clusters": self._has_skill_clusters,
            "model_version": self.model_version,
            "index_dir": str(self.index_dir),
        }

        # Cache stats
        stats["caches"] = {
            "query_cache_size": len(self._query_cache),
            "query_cache_max": self.QUERY_CACHE_SIZE,
            "result_cache_size": len(self._result_cache),
            "result_cache_max": self.RESULT_CACHE_SIZE,
        }

        # Index stats (if loaded)
        if self._has_vector_index:
            try:
                stats["index_stats"] = self.index_manager.get_stats()
            except Exception as e:
                stats["index_stats"] = {"error": str(e)}

        # Query expander stats
        if self.query_expander:
            stats["query_expander"] = self.query_expander.get_stats()

        # Database stats
        stats["embedding_stats"] = self.db.get_embedding_stats()

        return stats

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _apply_sql_filters(self, request: SearchRequest) -> list[dict]:
        """
        Apply SQL filters and return matching jobs.

        Args:
            request: Search request with filter parameters

        Returns:
            List of job dicts matching filters
        """
        return self.db.search_jobs(
            salary_min=request.salary_min,
            salary_max=request.salary_max,
            employment_type=request.employment_type,
            company_name=request.company,
            limit=100000,  # Get all matching for ranking
        )

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding for a query string (with caching).

        Args:
            query: Query text

        Returns:
            Embedding array of shape (dimension,)
        """
        # Check cache
        cache_key = f"query:{query}"
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        # Generate embedding
        embedding = self.generator.model.encode(
            query,
            normalize_embeddings=True,
        )
        embedding = np.asarray(embedding, dtype=np.float32)

        # Cache it
        self._query_cache[cache_key] = embedding

        return embedding

    def _compute_hybrid_scores(
        self,
        query: str,
        search_query: str,
        candidate_uuids: list[str],
        alpha: float,
        freshness_weight: float,
    ) -> list[tuple[str, float]]:
        """
        Compute hybrid scores combining semantic and keyword search.

        Formula: score = alpha * semantic + (1 - alpha) * bm25 + freshness_boost

        Args:
            query: Original query (for semantic embedding)
            search_query: Expanded query (for BM25)
            candidate_uuids: UUIDs to score
            alpha: Weight for semantic vs keyword (1.0 = semantic only)
            freshness_weight: Weight for freshness boost

        Returns:
            List of (uuid, score) tuples, sorted by score descending
        """
        candidate_set = set(candidate_uuids)

        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Get semantic scores via filtered FAISS search
        try:
            semantic_results = self.index_manager.search_jobs_filtered(
                query_embedding,
                allowed_uuids=candidate_set,
                k=len(candidate_uuids),
            )
            semantic_scores = {uuid: score for uuid, score in semantic_results}
        except IndexNotBuiltError:
            semantic_scores = {}

        # Get BM25 scores
        bm25_scores = self._get_bm25_scores(search_query, candidate_uuids)

        # Get freshness scores
        freshness_scores = (
            self._compute_freshness_scores(candidate_uuids)
            if freshness_weight > 0
            else {}
        )

        # Normalize scores to [0, 1] range for fair combination
        semantic_scores = self._normalize_scores(semantic_scores)
        bm25_scores = self._normalize_scores(bm25_scores)

        # Combine scores
        combined: dict[str, float] = {}
        for uuid in candidate_uuids:
            sem_score = semantic_scores.get(uuid, 0.0)
            bm25_score = bm25_scores.get(uuid, 0.0)
            fresh_score = freshness_scores.get(uuid, 0.5)

            combined[uuid] = (
                alpha * sem_score
                + (1 - alpha) * bm25_score
                + freshness_weight * fresh_score
            )

        # Sort by score descending
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def _get_bm25_scores(
        self, query: str, candidate_uuids: list[str]
    ) -> dict[str, float]:
        """
        Get BM25 scores for candidates.

        Uses filtered BM25 search that restricts scoring to the candidate set,
        ensuring no relevant candidate is missed due to global ranking cutoffs.

        Args:
            query: Search query
            candidate_uuids: UUIDs to score

        Returns:
            Dict mapping uuid -> BM25 score (higher = more relevant)
        """
        try:
            candidate_set = set(candidate_uuids)
            # BM25 returns negative scores (lower = better)
            bm25_results = self.db.bm25_search_filtered(query, candidate_set)

            # Negate to make higher = better
            scores = {}
            for uuid, score in bm25_results:
                scores[uuid] = -score

            return scores
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return {}

    def _keyword_only_scores(
        self,
        search_query: str,
        candidate_uuids: list[str],
        freshness_weight: float,
    ) -> list[tuple[str, float]]:
        """
        Compute scores using only keyword (BM25) search.

        Used when vector index is unavailable (degraded mode).

        Args:
            search_query: Search query
            candidate_uuids: UUIDs to score
            freshness_weight: Weight for freshness boost

        Returns:
            List of (uuid, score) tuples
        """
        bm25_scores = self._get_bm25_scores(search_query, candidate_uuids)
        bm25_scores = self._normalize_scores(bm25_scores)

        freshness_scores = (
            self._compute_freshness_scores(candidate_uuids)
            if freshness_weight > 0
            else {}
        )

        combined: dict[str, float] = {}
        for uuid in candidate_uuids:
            bm25_score = bm25_scores.get(uuid, 0.0)
            fresh_score = freshness_scores.get(uuid, 0.5)
            combined[uuid] = bm25_score + freshness_weight * fresh_score

        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def _keyword_fallback_search(
        self, request: SearchRequest, start_time: float
    ) -> SearchResponse:
        """
        Fallback search using only keywords when vectors fail.

        Args:
            request: Search request
            start_time: Time when search started

        Returns:
            SearchResponse with keyword-only results
        """
        candidates = self._apply_sql_filters(request)

        if not candidates:
            return SearchResponse(
                results=[],
                total_candidates=0,
                search_time_ms=(time.time() - start_time) * 1000,
                degraded=True,
            )

        scored_results = self._keyword_only_scores(
            search_query=request.query,
            candidate_uuids=[c["uuid"] for c in candidates],
            freshness_weight=request.freshness_weight,
        )

        filtered_results = scored_results[: request.limit]
        results = self._enrich_results(filtered_results)

        return SearchResponse(
            results=results,
            total_candidates=len(candidates),
            search_time_ms=(time.time() - start_time) * 1000,
            degraded=True,
        )

    def _compute_freshness_scores(self, uuids: list[str]) -> dict[str, float]:
        """
        Compute freshness scores based on posting date.

        More recent jobs get higher scores (0-1 range).

        Args:
            uuids: Job UUIDs to score

        Returns:
            Dict mapping uuid -> freshness score
        """
        scores: dict[str, float] = {}
        today = date.today()

        for uuid in uuids:
            job = self.db.get_job(uuid)
            if job and job.get("posted_date"):
                try:
                    posted = job["posted_date"]
                    if isinstance(posted, str):
                        posted = date.fromisoformat(posted)

                    days_ago = (today - posted).days
                    # Exponential decay: 30-day half-life
                    # Fresh jobs (0 days) = 1.0, 30 days = 0.5, 60 days = 0.25
                    scores[uuid] = np.exp(-days_ago / 30 * np.log(2))
                except (ValueError, TypeError):
                    scores[uuid] = 0.5
            else:
                scores[uuid] = 0.5

        return scores

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: Dict mapping uuid -> raw score

        Returns:
            Dict mapping uuid -> normalized score
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            # All scores are the same
            return {uuid: 0.5 for uuid in scores}

        return {
            uuid: (score - min_val) / (max_val - min_val)
            for uuid, score in scores.items()
        }

    def _enrich_results(
        self, scored_results: list[tuple[str, float]]
    ) -> list[JobResult]:
        """
        Convert (uuid, score) tuples to full JobResult objects.

        Args:
            scored_results: List of (uuid, score) tuples

        Returns:
            List of JobResult objects with full job data
        """
        results: list[JobResult] = []

        for uuid, score in scored_results:
            job = self.db.get_job(uuid)
            if job:
                # Parse posted_date if it's a string
                posted_date = job.get("posted_date")
                if isinstance(posted_date, str):
                    try:
                        posted_date = date.fromisoformat(posted_date)
                    except ValueError:
                        posted_date = None

                # Truncate description for response
                description = job.get("description", "") or ""
                if len(description) > 500:
                    description = description[:500] + "..."

                results.append(
                    JobResult(
                        uuid=job["uuid"],
                        title=job.get("title", ""),
                        company_name=job.get("company_name", ""),
                        description=description,
                        salary_min=job.get("salary_min"),
                        salary_max=job.get("salary_max"),
                        employment_type=job.get("employment_type"),
                        skills=job.get("skills"),
                        location=job.get("location"),
                        posted_date=posted_date,
                        job_url=job.get("job_url"),
                        similarity_score=score,
                    )
                )

        return results

    def _log_search(self, request: SearchRequest, response: SearchResponse) -> None:
        """
        Log search to analytics table.

        Args:
            request: Search request
            response: Search response
        """
        try:
            query_type = "hybrid" if not self._degraded else "keyword"

            self.db.log_search(
                query=request.query,
                query_type=query_type,
                result_count=len(response.results),
                latency_ms=response.search_time_ms,
                cache_hit=response.cache_hit,
                degraded=response.degraded,
                filters_used={
                    "salary_min": request.salary_min,
                    "salary_max": request.salary_max,
                    "employment_type": request.employment_type,
                    "company": request.company,
                    "region": request.region,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to log search analytics: {e}")

    def get_skill_cloud(self, min_jobs: int = 10, limit: int = 100) -> dict:
        """
        Get skill frequency data for visualization (word cloud, bar chart).

        Fetches skill occurrence counts from the database and annotates
        each skill with its cluster ID from the query expander (if loaded).

        Args:
            min_jobs: Minimum job count for a skill to be included
            limit: Maximum number of skills to return

        Returns:
            Dict with 'items' (list of skill dicts) and 'total_unique_skills'
        """
        if not self._loaded:
            self.load()

        # Single DB call — get all skills, then filter in Python
        all_skills = self.db.get_skill_frequencies(min_jobs=1, limit=100000)
        total_unique = len(all_skills)

        # Apply caller's min_jobs filter (already sorted by freq desc)
        filtered = [(s, c) for s, c in all_skills if c >= min_jobs][:limit]

        items = []
        for skill, count in filtered:
            cluster_id = None
            if self.query_expander:
                cluster_id = self.query_expander.skill_to_cluster.get(skill)
            items.append({
                "skill": skill,
                "job_count": count,
                "cluster_id": cluster_id,
            })

        return {
            "items": items,
            "total_unique_skills": total_unique,
        }

    def get_related_skills(self, skill: str, k: int = 10) -> Optional[dict]:
        """
        Get skills related to a given skill.

        Primary strategy: FAISS embedding similarity on the skill index,
        annotated with cluster membership from QueryExpander.
        Fallback: cluster-only lookup when FAISS skill index is unavailable.

        Args:
            skill: Skill name to find relatives for
            k: Maximum number of related skills to return

        Returns:
            Dict with 'skill' and 'related' list, or None if skill not found
        """
        if not self._loaded:
            self.load()

        source_cluster = None
        if self.query_expander:
            source_cluster = self.query_expander.skill_to_cluster.get(skill)

        # Primary: FAISS embedding similarity
        if (
            self._has_vector_index
            and "skills" in self.index_manager.indexes
            and self.index_manager.indexes["skills"] is not None
        ):
            skill_idx = self.index_manager.skill_to_idx.get(skill)
            if skill_idx is not None:
                # Reconstruct the skill's own embedding from the Flat index
                skill_embedding = self.index_manager.indexes["skills"].reconstruct(
                    skill_idx
                )
                similar = self.index_manager.search_skills(skill_embedding, k=k + 1)

                related = []
                for s, score in similar:
                    if s == skill:
                        continue
                    same_cluster = False
                    if self.query_expander:
                        s_cluster = self.query_expander.skill_to_cluster.get(s)
                        same_cluster = (
                            s_cluster is not None and s_cluster == source_cluster
                        )
                    related.append({
                        "skill": s,
                        "similarity": round(float(score), 4),
                        "same_cluster": same_cluster,
                    })

                return {
                    "skill": skill,
                    "related": related[:k],
                }

        # Fallback: cluster-only (no similarity scores available)
        if self.query_expander:
            cluster_skills = self.query_expander.get_related_skills(skill, k=k)
            if cluster_skills:
                return {
                    "skill": skill,
                    "related": [
                        {"skill": s, "similarity": 1.0, "same_cluster": True}
                        for s in cluster_skills
                    ],
                }

        return None  # Skill not found

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._query_cache.clear()
        self._result_cache.clear()
        logger.info("Search engine caches cleared")
