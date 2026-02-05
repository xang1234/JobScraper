"""
Embedding generator for semantic search.

Transforms job text into vector embeddings using Sentence Transformers.
Handles batch processing, skill clustering, and company multi-centroid generation.
"""

import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from ..database import MCFDatabase
    from ..models import Job

from .models import EmbeddingStats, SkillClusterResult

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates semantic embeddings for jobs, skills, and companies.

    Uses Sentence Transformers with all-MiniLM-L6-v2 model (384 dimensions).
    Designed for batch processing with progress tracking.

    Handles:
    - Job embedding generation (composite text strategy)
    - Skill extraction and clustering for query expansion
    - Company multi-centroid embedding preparation

    Example:
        generator = EmbeddingGenerator()
        embedding = generator.generate_job_embedding(job)

        # Batch processing with progress tracking
        stats = generator.generate_all(db, progress_callback=print)
        print(f"Processed {stats.jobs_processed} jobs")
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSION = 384

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize generator with lazy model loading.

        Args:
            model_name: Override default model (e.g., for testing)
            device: 'cpu', 'cuda', 'mps', or None for auto-detect
        """
        self._model: Optional["SentenceTransformer"] = None
        self.model_name = model_name or self.MODEL_NAME
        self.device = device

    @property
    def model(self) -> "SentenceTransformer":
        """
        Lazy load the model on first use.

        This defers the 2-3 second loading time until embeddings are actually needed.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded on device: {self._model.device}")
        return self._model

    def _compose_job_text(self, job: "Job") -> str:
        """
        Create composite text for embedding.

        Title is included twice for 2x weighting - it's the most discriminative
        signal (e.g., "Data Scientist" vs "Software Engineer" is obvious from title).
        Description is truncated to avoid dominating the embedding.

        Args:
            job: Job object to compose text from

        Returns:
            Composite text string for embedding
        """
        parts = [
            job.title,  # Include title once
            job.title,  # Repeat for 2x weight
            job.description_text[:500] if job.description_text else "",
            job.skills_list,
            job.categories_list,
        ]
        return " ".join(filter(None, parts))

    def generate_job_embedding(self, job: "Job") -> np.ndarray:
        """
        Generate embedding for a single job.

        Args:
            job: Job object to embed

        Returns:
            Normalized embedding array of shape (DIMENSION,)
        """
        text = self._compose_job_text(job)
        embedding = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(embedding, dtype=np.float32)

    def generate_job_embeddings_batch(
        self,
        jobs: list["Job"],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple jobs efficiently.

        Uses batched encoding for better GPU utilization.

        Args:
            jobs: List of Job objects
            batch_size: Encoding batch size (affects memory usage)
            show_progress: Show progress bar

        Returns:
            Embedding matrix of shape (n_jobs, DIMENSION)
        """
        texts = [self._compose_job_text(job) for job in jobs]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def generate_skill_embedding(self, skill: str) -> np.ndarray:
        """
        Generate embedding for a single skill.

        Args:
            skill: Skill name string

        Returns:
            Normalized embedding array of shape (DIMENSION,)
        """
        embedding = self.model.encode(skill, normalize_embeddings=True)
        return np.asarray(embedding, dtype=np.float32)

    def generate_skill_embeddings_batch(
        self,
        skills: list[str],
        batch_size: int = 64,
    ) -> dict[str, np.ndarray]:
        """
        Generate embeddings for multiple skills.

        Args:
            skills: List of skill names
            batch_size: Encoding batch size

        Returns:
            Dict mapping skill name -> embedding array
        """
        if not skills:
            return {}

        embeddings = self.model.encode(
            skills,
            normalize_embeddings=True,
            batch_size=batch_size,
        )

        return {
            skill: np.asarray(emb, dtype=np.float32)
            for skill, emb in zip(skills, embeddings)
        }

    def cluster_skills(
        self,
        skills: list[str],
        n_clusters: Optional[int] = None,
    ) -> SkillClusterResult:
        """
        Cluster skills by embedding similarity for query expansion.

        Uses agglomerative clustering with cosine distance, which produces
        more semantically coherent clusters than K-means for text.

        This enables query expansion: "ML" -> ["ML", "Machine Learning", "Deep Learning"]

        Args:
            skills: List of skill names to cluster
            n_clusters: Target number of clusters. If None, auto-computed (~5 skills per cluster)

        Returns:
            SkillClusterResult with clusters, mappings, and centroids
        """
        if len(skills) < 3:
            # Too few skills to cluster meaningfully
            return SkillClusterResult(
                clusters={0: skills},
                skill_to_cluster={s: 0 for s in skills},
                cluster_centroids={},
            )

        from sklearn.cluster import AgglomerativeClustering

        # Generate embeddings for all skills
        logger.info(f"Generating embeddings for {len(skills)} skills...")
        embeddings = np.array(
            [self.generate_skill_embedding(s) for s in skills],
            dtype=np.float32,
        )

        # Normalize for cosine distance (embeddings are already normalized, but be safe)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms

        # Auto-compute n_clusters if not specified (~5 skills per cluster)
        if n_clusters is None:
            n_clusters = max(1, min(len(skills) // 5, 100))

        logger.info(f"Clustering {len(skills)} skills into {n_clusters} clusters...")

        # Agglomerative clustering with cosine distance
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(normalized)

        # Build result structures
        clusters: dict[int, list[str]] = defaultdict(list)
        skill_to_cluster: dict[str, int] = {}
        cluster_centroids: dict[int, list[float]] = {}

        for skill, label in zip(skills, labels):
            cluster_id = int(label)
            clusters[cluster_id].append(skill)
            skill_to_cluster[skill] = cluster_id

        # Compute cluster centroids
        for cluster_id, cluster_skills in clusters.items():
            indices = [skills.index(s) for s in cluster_skills]
            centroid = embeddings[indices].mean(axis=0)
            # Normalize centroid
            centroid = centroid / np.linalg.norm(centroid)
            cluster_centroids[cluster_id] = centroid.tolist()

        logger.info(f"Created {len(clusters)} skill clusters")

        return SkillClusterResult(
            clusters=dict(clusters),
            skill_to_cluster=skill_to_cluster,
            cluster_centroids=cluster_centroids,
        )

    def _compute_recency_weights(self, jobs: list["Job"]) -> np.ndarray:
        """
        Compute recency weights for jobs (more recent = higher weight).

        Used for company centroid calculation to emphasize recent hiring patterns.

        Args:
            jobs: List of Job objects

        Returns:
            Weight array of shape (n_jobs,), summing to 1.0
        """
        weights = []
        for job in jobs:
            if job.posted_date:
                # Days since posting (capped at 365)
                from datetime import date

                days_ago = min((date.today() - job.posted_date).days, 365)
                # Exponential decay: recent jobs weighted higher
                weight = np.exp(-days_ago / 90)  # 90-day half-life
            else:
                weight = 0.5  # Default weight for jobs without dates
            weights.append(weight)

        weights = np.array(weights, dtype=np.float32)
        # Normalize to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(jobs), dtype=np.float32) / len(jobs)

        return weights

    def generate_company_embeddings(
        self,
        company_jobs: dict[str, list["Job"]],
        k_centroids: int = 3,
    ) -> dict[str, list[np.ndarray]]:
        """
        Generate multi-centroid embeddings for companies.

        For companies with >= 10 jobs: K-means clustering into job families
        For companies with < 10 jobs: Single weighted centroid (recent jobs weighted higher)

        Multi-centroid representation captures different job types a company hires for
        (e.g., Google hires both engineers and sales roles - one centroid can't capture both).

        Args:
            company_jobs: Dict mapping company_name -> list of Job objects
            k_centroids: Number of centroids for large companies

        Returns:
            Dict mapping company_name -> list of centroid embeddings
        """
        from sklearn.cluster import KMeans

        company_centroids: dict[str, list[np.ndarray]] = {}

        for company, jobs in company_jobs.items():
            if not jobs:
                continue

            # Get job embeddings
            job_embeddings = self.generate_job_embeddings_batch(jobs)

            if len(jobs) < 10:
                # Single weighted centroid for small companies
                weights = self._compute_recency_weights(jobs)
                centroid = np.average(job_embeddings, weights=weights, axis=0)
                # Normalize
                centroid = centroid / np.linalg.norm(centroid)
                company_centroids[company] = [centroid.astype(np.float32)]
            else:
                # K-means clustering for large companies
                actual_k = min(k_centroids, len(jobs) // 3)
                actual_k = max(1, actual_k)

                kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
                kmeans.fit(job_embeddings)

                # Normalize centroids
                centroids = []
                for centroid in kmeans.cluster_centers_:
                    centroid = centroid / np.linalg.norm(centroid)
                    centroids.append(centroid.astype(np.float32))

                company_centroids[company] = centroids

        return company_centroids

    def _save_skill_clusters(
        self,
        cluster_result: SkillClusterResult,
        output_dir: Path = Path("data/embeddings"),
    ) -> None:
        """
        Save skill cluster data to disk for QueryExpander.

        Creates pickle files that can be loaded by the search module
        for query expansion.

        Args:
            cluster_result: Result from cluster_skills()
            output_dir: Directory to save cluster files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "skill_clusters.pkl", "wb") as f:
            pickle.dump(cluster_result.clusters, f)

        with open(output_dir / "skill_to_cluster.pkl", "wb") as f:
            pickle.dump(cluster_result.skill_to_cluster, f)

        with open(output_dir / "skill_cluster_centroids.pkl", "wb") as f:
            pickle.dump(cluster_result.cluster_centroids, f)

        logger.info(f"Saved skill cluster data to {output_dir}")

    def generate_all(
        self,
        db: "MCFDatabase",
        batch_size: int = 32,
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[EmbeddingStats], None]] = None,
        output_dir: Path = Path("data/embeddings"),
    ) -> EmbeddingStats:
        """
        Generate all embeddings: jobs, skills (with clustering), companies.

        This is the main entry point for batch embedding generation.
        Includes skill clustering for query expansion support.

        Args:
            db: MCFDatabase instance
            batch_size: Number of jobs to process per batch
            skip_existing: Skip jobs that already have embeddings
            progress_callback: Called with stats after each batch
            output_dir: Directory for skill cluster output files

        Returns:
            EmbeddingStats with processing metrics
        """
        from datetime import datetime

        stats = EmbeddingStats(started_at=datetime.now())
        start_time = time.time()

        # Step 1: Count jobs to process
        if skip_existing:
            jobs_to_process = db.get_jobs_without_embeddings(limit=1000000)
            stats.jobs_total = len(jobs_to_process)
        else:
            stats.jobs_total = db.count_jobs()
            jobs_to_process = None

        logger.info(f"Starting embedding generation for {stats.jobs_total} jobs")

        # Step 2: Generate job embeddings in batches
        if jobs_to_process is None:
            # Process all jobs (need to iterate through them)
            all_uuids = list(db.get_all_uuids())
            stats.jobs_total = len(all_uuids)

            for i in range(0, len(all_uuids), batch_size):
                batch_uuids = all_uuids[i : i + batch_size]
                self._process_job_batch_by_uuids(db, batch_uuids, stats)

                stats.elapsed_seconds = time.time() - start_time
                if progress_callback:
                    progress_callback(stats)
        else:
            # Process jobs that need embeddings
            for i in range(0, len(jobs_to_process), batch_size):
                batch = jobs_to_process[i : i + batch_size]
                self._process_job_batch(db, batch, stats)

                stats.elapsed_seconds = time.time() - start_time
                if progress_callback:
                    progress_callback(stats)

        # Step 3: Extract and cluster skills
        logger.info("Extracting and clustering skills...")
        skills = db.get_all_unique_skills()
        stats.unique_skills = len(skills)

        if skills:
            cluster_result = self.cluster_skills(skills)
            stats.skill_clusters = cluster_result.num_clusters

            # Store skill embeddings in database
            logger.info(f"Storing {len(skills)} skill embeddings...")
            skill_embeddings = self.generate_skill_embeddings_batch(skills)
            for skill, embedding in skill_embeddings.items():
                db.upsert_embedding(skill, "skill", embedding, self.model_name)

            # Save cluster data to disk for QueryExpander
            self._save_skill_clusters(cluster_result, output_dir)

        # Step 4: Finalize stats
        stats.elapsed_seconds = time.time() - start_time
        stats.completed_at = datetime.now()

        logger.info(
            f"Embedding generation complete: {stats.jobs_processed} jobs, "
            f"{stats.unique_skills} skills, {stats.skill_clusters} clusters "
            f"in {stats.elapsed_seconds:.1f}s ({stats.jobs_per_second:.1f} jobs/sec)"
        )

        return stats

    def _process_job_batch(
        self,
        db: "MCFDatabase",
        jobs_data: list[dict],
        stats: EmbeddingStats,
    ) -> None:
        """
        Process a batch of jobs from get_jobs_without_embeddings result.

        Args:
            db: Database instance
            jobs_data: List of job dicts with uuid, title, description, skills
            stats: Stats object to update
        """
        from ..models import Job

        # Convert dict data to minimal Job-like objects for embedding
        # We need: title, description_text, skills_list, categories_list
        texts = []
        uuids = []

        for job_dict in jobs_data:
            try:
                # Compose text from dict data
                title = job_dict.get("title", "")
                description = job_dict.get("description", "")[:500]
                skills = job_dict.get("skills", "")

                # Simple text composition (title 2x, description, skills)
                text = f"{title} {title} {description} {skills}"
                texts.append(text)
                uuids.append(job_dict["uuid"])
            except Exception as e:
                logger.warning(f"Failed to process job {job_dict.get('uuid')}: {e}")
                stats.jobs_failed += 1

        if not texts:
            return

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=len(texts),
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Store in database
        db.batch_upsert_embeddings(uuids, "job", embeddings, self.model_name)
        stats.jobs_processed += len(uuids)

    def _process_job_batch_by_uuids(
        self,
        db: "MCFDatabase",
        uuids: list[str],
        stats: EmbeddingStats,
    ) -> None:
        """
        Process a batch of jobs by their UUIDs.

        Args:
            db: Database instance
            uuids: List of job UUIDs to process
            stats: Stats object to update
        """
        texts = []
        valid_uuids = []

        for uuid in uuids:
            job_data = db.get_job(uuid)
            if job_data is None:
                stats.jobs_skipped += 1
                continue

            try:
                title = job_data.get("title", "")
                description = (job_data.get("description") or "")[:500]
                skills = job_data.get("skills", "")

                text = f"{title} {title} {description} {skills}"
                texts.append(text)
                valid_uuids.append(uuid)
            except Exception as e:
                logger.warning(f"Failed to process job {uuid}: {e}")
                stats.jobs_failed += 1

        if not texts:
            return

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=len(texts),
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Store in database
        db.batch_upsert_embeddings(valid_uuids, "job", embeddings, self.model_name)
        stats.jobs_processed += len(valid_uuids)
