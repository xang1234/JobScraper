"""
FAISS Index Manager for semantic search.

Manages FAISS indexes for efficient similarity search across jobs, skills, and companies.
Supports three index types optimized for different use cases:

- Jobs: IVFFlat index for large-scale (100K+) vector search with clustering
- Skills: Flat index for exact search on smaller (~1K) skill embeddings
- Companies: Multi-centroid flat index for diverse company representations

The IVFFlat index partitions vectors into clusters (Voronoi cells) and only searches
nearby clusters at query time. This trades off some recall for significant speed gains
on large datasets.

Example:
    manager = FAISSIndexManager(index_dir=Path("data/embeddings"))

    # Build index from embeddings
    manager.build_job_index(embeddings, uuids)

    # Search for similar jobs
    results = manager.search_jobs(query_vector, k=10)

    # Persist to disk
    manager.save()
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class IndexNotBuiltError(Exception):
    """Raised when attempting to search an index that hasn't been built."""

    pass


class IndexCompatibilityError(Exception):
    """Raised when model version doesn't match the index."""

    pass


class FAISSIndexManager:
    """
    Manages FAISS indexes for semantic search.

    Handles three index types:
    - jobs: IVFFlat index for job embeddings (optimized for large datasets)
    - skills: Flat index for skill embeddings (exact search, ~1K items)
    - companies: Flat index for company centroids (multi-centroid per company)

    The manager handles:
    - Index building with automatic nlist tuning
    - Efficient search with configurable nprobe
    - Filtered search for pre-filtered candidate sets
    - Persistence to/from disk
    - Model version compatibility checking

    Example:
        manager = FAISSIndexManager(index_dir=Path("data/embeddings"))
        manager.build_job_index(embeddings, uuids)
        results = manager.search_jobs(query_vector, k=10)
        manager.save()
    """

    # Default embedding dimension (all-MiniLM-L6-v2)
    DIMENSION = 384

    # Threshold for using temporary index vs filter-after-search
    FILTERED_SEARCH_THRESHOLD = 10000

    def __init__(
        self,
        index_dir: Path = Path("data/embeddings"),
        model_version: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize index manager.

        Args:
            index_dir: Directory for index files
            model_version: Embedding model version for compatibility checking
        """
        self.index_dir = Path(index_dir)
        self.model_version = model_version

        # Index storage
        self.indexes: dict[str, "faiss.Index"] = {}

        # Mapping storage (index position -> entity ID)
        self.uuid_to_idx: dict[str, int] = {}
        self.idx_to_uuid: dict[int, str] = {}

        self.skill_names: list[str] = []
        self.skill_to_idx: dict[str, int] = {}

        # Company multi-centroid mappings
        self.company_names: list[str] = []
        self._company_centroid_map: dict[str, list[int]] = {}  # company -> centroid indices
        self._all_company_centroids: Optional[np.ndarray] = None

        # Index metadata
        self._nlist: int = 0  # Number of clusters for IVFFlat
        self._default_nprobe: int = 10  # Default clusters to search

    # =========================================================================
    # Build Methods
    # =========================================================================

    def build_job_index(
        self,
        embeddings: np.ndarray,
        uuids: list[str],
        nlist: Optional[int] = None,
    ) -> None:
        """
        Build IVFFlat index for job embeddings.

        IVFFlat creates nlist clusters (Voronoi cells) and assigns each vector
        to its nearest cluster. At search time, only nprobe nearest clusters
        are searched, trading recall for speed.

        Args:
            embeddings: Job embeddings matrix of shape (n_jobs, 384)
            uuids: List of job UUIDs in same order as embeddings
            nlist: Number of clusters (auto-calculated if None)

        Raises:
            ValueError: If embeddings and uuids have different lengths
        """
        import faiss

        if len(embeddings) != len(uuids):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and uuids ({len(uuids)}) must have same length"
            )

        n_vectors = len(embeddings)
        dimension = embeddings.shape[1]

        logger.info(f"Building job index with {n_vectors} vectors, dimension {dimension}")

        # Ensure float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # Calculate optimal nlist if not provided
        if nlist is None:
            nlist = self._calculate_nlist(n_vectors)

        self._nlist = nlist
        self._default_nprobe = self._calculate_nprobe(nlist)

        logger.info(f"Using nlist={nlist}, default nprobe={self._default_nprobe}")

        # For small datasets, use Flat index (no clustering needed)
        if n_vectors < 1000:
            logger.info("Small dataset, using Flat index instead of IVFFlat")
            index = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized vectors
            index.add(embeddings)
        else:
            # Build IVFFlat index
            # Quantizer: Flat index for cluster centroids
            quantizer = faiss.IndexFlatIP(dimension)

            # IVFFlat: Inverted file with flat storage
            # METRIC_INNER_PRODUCT for normalized vectors = cosine similarity
            index = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )

            # Train the index (learn cluster centroids)
            logger.info("Training IVFFlat index...")
            index.train(embeddings)

            # Add vectors
            logger.info("Adding vectors to index...")
            index.add(embeddings)

            # Enable DirectMap for reconstruction (needed for filtered search)
            # This creates ID -> (list_id, offset) mapping for O(1) lookup
            logger.info("Building direct map for reconstruction...")
            index.make_direct_map()

        self.indexes["jobs"] = index

        # Build UUID mappings
        self.uuid_to_idx = {uuid: idx for idx, uuid in enumerate(uuids)}
        self.idx_to_uuid = {idx: uuid for idx, uuid in enumerate(uuids)}

        logger.info(f"Job index built: {index.ntotal} vectors indexed")

    def build_skill_index(
        self,
        embeddings: np.ndarray,
        skill_names: list[str],
    ) -> None:
        """
        Build Flat index for skill embeddings.

        Uses exact (brute-force) search since skill sets are typically small (~1K).
        Flat indexes have 100% recall and are simpler to maintain.

        Args:
            embeddings: Skill embeddings matrix of shape (n_skills, 384)
            skill_names: List of skill names in same order as embeddings

        Raises:
            ValueError: If embeddings and skill_names have different lengths
        """
        import faiss

        if len(embeddings) != len(skill_names):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and skill_names ({len(skill_names)}) "
                "must have same length"
            )

        n_skills = len(embeddings)
        dimension = embeddings.shape[1]

        logger.info(f"Building skill index with {n_skills} skills")

        # Ensure float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # Simple Flat index for exact search
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        self.indexes["skills"] = index
        self.skill_names = list(skill_names)
        self.skill_to_idx = {name: idx for idx, name in enumerate(skill_names)}

        logger.info(f"Skill index built: {index.ntotal} skills indexed")

    def build_company_index(
        self,
        company_centroids: dict[str, list[np.ndarray]],
    ) -> None:
        """
        Build index for company multi-centroids.

        Each company can have multiple centroids representing different job families
        (e.g., engineering roles vs. business roles). During search, similarity is
        computed as max(query · centroid) across all centroids for each company.

        Args:
            company_centroids: Dict mapping company_name -> list of centroid arrays
        """
        import faiss

        logger.info(f"Building company index with {len(company_centroids)} companies")

        # Flatten all centroids into single array, tracking company ownership
        all_centroids = []
        company_names = []
        centroid_map: dict[str, list[int]] = {}

        centroid_idx = 0
        for company, centroids in company_centroids.items():
            centroid_indices = []
            for centroid in centroids:
                all_centroids.append(centroid.astype(np.float32))
                centroid_indices.append(centroid_idx)
                centroid_idx += 1

            centroid_map[company] = centroid_indices
            company_names.append(company)

        if not all_centroids:
            logger.warning("No company centroids provided, skipping company index")
            return

        # Stack into matrix
        centroids_matrix = np.vstack(all_centroids).astype(np.float32)
        centroids_matrix = np.ascontiguousarray(centroids_matrix)

        logger.info(f"Total centroids: {len(all_centroids)} for {len(company_names)} companies")

        # Build Flat index
        dimension = centroids_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(centroids_matrix)

        self.indexes["companies"] = index
        self.company_names = company_names
        self._company_centroid_map = centroid_map
        self._all_company_centroids = centroids_matrix

        logger.info(f"Company index built: {index.ntotal} centroids indexed")

    # =========================================================================
    # Search Methods
    # =========================================================================

    def search_jobs(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """
        Search for similar jobs.

        For IVFFlat indexes, nprobe controls recall/speed tradeoff:
        - Higher nprobe = more clusters searched = better recall, slower
        - Lower nprobe = fewer clusters = faster, may miss some results

        Args:
            query_vector: Query embedding of shape (384,)
            k: Number of results to return
            nprobe: Clusters to search (uses default if None)

        Returns:
            List of (uuid, similarity_score) tuples, sorted by similarity descending

        Raises:
            IndexNotBuiltError: If job index hasn't been built
        """
        if "jobs" not in self.indexes or self.indexes["jobs"] is None:
            raise IndexNotBuiltError(
                "Job index not built. Run embed-generate and build-index first."
            )

        import faiss

        index = self.indexes["jobs"]

        # Set nprobe for IVFFlat indexes
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe or self._default_nprobe

        # Prepare query vector
        query = np.ascontiguousarray(
            query_vector.reshape(1, -1).astype(np.float32)
        )

        # Search
        similarities, indices = index.search(query, k)

        # Build results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and idx in self.idx_to_uuid:
                results.append((self.idx_to_uuid[idx], float(sim)))

        return results

    def search_jobs_filtered(
        self,
        query_vector: np.ndarray,
        allowed_uuids: set[str],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Search only among specific UUIDs (for filter-then-search).

        This is used when SQL filters have already narrowed down candidates.
        The strategy depends on the number of candidates:
        - If < 10K: Build temporary Flat index (faster for small sets)
        - If >= 10K: Search full index and filter results

        Args:
            query_vector: Query embedding of shape (384,)
            allowed_uuids: Set of UUIDs to search within
            k: Number of results to return

        Returns:
            List of (uuid, similarity_score) tuples, sorted by similarity descending

        Raises:
            IndexNotBuiltError: If job index hasn't been built
        """
        if "jobs" not in self.indexes or self.indexes["jobs"] is None:
            raise IndexNotBuiltError(
                "Job index not built. Run embed-generate and build-index first."
            )

        if not allowed_uuids:
            return []

        # Choose strategy based on candidate set size
        if len(allowed_uuids) < self.FILTERED_SEARCH_THRESHOLD:
            return self._search_with_temp_index(query_vector, allowed_uuids, k)
        else:
            return self._search_and_filter(query_vector, allowed_uuids, k)

    def _search_with_temp_index(
        self,
        query_vector: np.ndarray,
        allowed_uuids: set[str],
        k: int,
    ) -> list[tuple[str, float]]:
        """
        Build temporary Flat index for small candidate sets.

        For small sets, it's faster to:
        1. Reconstruct embeddings for allowed UUIDs
        2. Build a temporary brute-force index
        3. Search the small index

        This avoids searching large clusters that mostly contain irrelevant results.
        """
        import faiss

        # Collect embeddings for allowed UUIDs
        embeddings = []
        uuids = []

        for uuid in allowed_uuids:
            if uuid in self.uuid_to_idx:
                idx = self.uuid_to_idx[uuid]
                # Reconstruct embedding from index
                embedding = self.indexes["jobs"].reconstruct(idx)
                embeddings.append(embedding)
                uuids.append(uuid)

        if not embeddings:
            return []

        # Build temporary Flat index
        embeddings_matrix = np.vstack(embeddings).astype(np.float32)
        temp_index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
        temp_index.add(embeddings_matrix)

        # Search
        query = np.ascontiguousarray(
            query_vector.reshape(1, -1).astype(np.float32)
        )
        similarities, indices = temp_index.search(query, min(k, len(uuids)))

        # Build results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0:
                results.append((uuids[idx], float(sim)))

        return results

    def _search_and_filter(
        self,
        query_vector: np.ndarray,
        allowed_uuids: set[str],
        k: int,
    ) -> list[tuple[str, float]]:
        """
        Search full index and filter results.

        For large candidate sets, it's more efficient to:
        1. Search the full index with a larger k
        2. Filter results to only include allowed UUIDs
        3. Return top k from filtered results

        We search for more results than needed to account for filtering.
        """
        # Search with larger k to account for filtering
        # Factor: expected fraction of results that will be allowed
        coverage = len(allowed_uuids) / max(1, self.indexes["jobs"].ntotal)
        search_k = min(
            int(k / max(coverage, 0.01)) + 100,  # At least k/coverage + buffer
            self.indexes["jobs"].ntotal,  # Can't exceed total
        )

        # Search full index
        all_results = self.search_jobs(query_vector, k=search_k)

        # Filter to allowed UUIDs
        filtered = [
            (uuid, score) for uuid, score in all_results
            if uuid in allowed_uuids
        ]

        return filtered[:k]

    def search_skills(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Search for similar skills.

        Uses the skill Flat index for exact similarity search.

        Args:
            query_vector: Query embedding of shape (384,)
            k: Number of results to return

        Returns:
            List of (skill_name, similarity_score) tuples

        Raises:
            IndexNotBuiltError: If skill index hasn't been built
        """
        if "skills" not in self.indexes or self.indexes["skills"] is None:
            raise IndexNotBuiltError(
                "Skill index not built. Run embed-generate first."
            )

        index = self.indexes["skills"]

        # Prepare query
        query = np.ascontiguousarray(
            query_vector.reshape(1, -1).astype(np.float32)
        )

        # Search
        similarities, indices = index.search(query, k)

        # Build results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and idx < len(self.skill_names):
                results.append((self.skill_names[idx], float(sim)))

        return results

    def search_companies(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Search for similar companies.

        For each company, we have multiple centroids representing different
        job families. Similarity = max(query · centroid) across all centroids.

        This captures companies with diverse hiring needs (e.g., Google hires
        both engineers and salespeople - a query for "engineer" should match
        Google based on their engineering centroid).

        Args:
            query_vector: Query embedding of shape (384,)
            k: Number of results to return

        Returns:
            List of (company_name, max_similarity_score) tuples

        Raises:
            IndexNotBuiltError: If company index hasn't been built
        """
        if "companies" not in self.indexes or self.indexes["companies"] is None:
            raise IndexNotBuiltError(
                "Company index not built. Run embed-generate first."
            )

        if self._all_company_centroids is None or len(self._company_centroid_map) == 0:
            return []

        # Compute similarities to all centroids at once
        query = query_vector.reshape(1, -1).astype(np.float32)
        all_similarities = np.dot(self._all_company_centroids, query.T).flatten()

        # For each company, take max similarity across their centroids
        company_scores: dict[str, float] = {}
        for company, centroid_indices in self._company_centroid_map.items():
            max_sim = max(all_similarities[idx] for idx in centroid_indices)
            company_scores[company] = float(max_sim)

        # Sort and return top k
        sorted_companies = sorted(
            company_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_companies[:k]

    def get_company_centroids(self, company_name: str) -> Optional[np.ndarray]:
        """
        Get pre-computed centroids for a specific company.

        Returns the centroid vectors stored in the company index for the
        given company. Used by search_engine to compare companies via
        multi-centroid matching.

        Args:
            company_name: Company name to look up

        Returns:
            Array of shape (n_centroids, dimension) or None if company not found
        """
        if self._all_company_centroids is None or company_name not in self._company_centroid_map:
            return None

        indices = self._company_centroid_map[company_name]
        return self._all_company_centroids[indices]

    def has_company_index(self) -> bool:
        """Check if a company index is loaded and available."""
        return (
            "companies" in self.indexes
            and self.indexes["companies"] is not None
            and self._all_company_centroids is not None
            and len(self._company_centroid_map) > 0
        )

    # =========================================================================
    # Update Methods
    # =========================================================================

    def add_jobs(
        self,
        embeddings: np.ndarray,
        uuids: list[str],
    ) -> None:
        """
        Add new jobs to existing index (incremental update).

        For IVFFlat indexes, new vectors are assigned to existing clusters.
        The cluster centroids don't change, which may reduce quality if
        many new vectors are added. Consider rebuilding periodically.

        Args:
            embeddings: New job embeddings of shape (n_new, 384)
            uuids: UUIDs for new jobs

        Raises:
            IndexNotBuiltError: If job index hasn't been built
            ValueError: If embeddings and uuids have different lengths
        """
        if "jobs" not in self.indexes or self.indexes["jobs"] is None:
            raise IndexNotBuiltError("Cannot add to unbuilt index")

        if len(embeddings) != len(uuids):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and uuids ({len(uuids)}) "
                "must have same length"
            )

        # Get current max index
        current_max = max(self.idx_to_uuid.keys()) if self.idx_to_uuid else -1

        # Add to index
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self.indexes["jobs"].add(embeddings)

        # Update mappings
        for i, uuid in enumerate(uuids):
            new_idx = current_max + 1 + i
            self.uuid_to_idx[uuid] = new_idx
            self.idx_to_uuid[new_idx] = uuid

        logger.info(f"Added {len(uuids)} jobs to index (total: {self.indexes['jobs'].ntotal})")

    def remove_jobs(self, uuids: list[str]) -> None:
        """
        Remove jobs from index.

        Note: IVFFlat indexes don't support direct removal. This method
        marks the UUIDs as removed in the mapping, but the vectors remain
        in the index until a full rebuild.

        For true removal, call rebuild_job_index() after removing.

        Args:
            uuids: UUIDs to remove
        """
        removed_count = 0
        for uuid in uuids:
            if uuid in self.uuid_to_idx:
                idx = self.uuid_to_idx[uuid]
                del self.uuid_to_idx[uuid]
                del self.idx_to_uuid[idx]
                removed_count += 1

        if removed_count > 0:
            logger.warning(
                f"Marked {removed_count} jobs as removed. "
                "Call rebuild_job_index() to reclaim space."
            )

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self) -> None:
        """
        Save all indexes and mappings to disk.

        Creates the following files in index_dir:
        - jobs.index: FAISS job index
        - jobs_uuids.npy: UUID list in index order
        - skills.index: FAISS skill index
        - skills_names.pkl: Skill name list
        - companies.index: FAISS company centroids index
        - companies_centroids.pkl: Company centroid mappings
        - companies_names.pkl: Company name list
        - index_metadata.pkl: nlist, nprobe, model version
        """
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save job index
        if "jobs" in self.indexes and self.indexes["jobs"] is not None:
            faiss.write_index(
                self.indexes["jobs"],
                str(self.index_dir / "jobs.index"),
            )

            # Save UUID mappings as ordered list
            ordered_uuids = [
                self.idx_to_uuid[i]
                for i in range(len(self.idx_to_uuid))
            ]
            np.save(self.index_dir / "jobs_uuids.npy", ordered_uuids)

            logger.info(f"Saved job index with {len(ordered_uuids)} vectors")

        # Save skill index
        if "skills" in self.indexes and self.indexes["skills"] is not None:
            faiss.write_index(
                self.indexes["skills"],
                str(self.index_dir / "skills.index"),
            )

            with open(self.index_dir / "skills_names.pkl", "wb") as f:
                pickle.dump(self.skill_names, f)

            logger.info(f"Saved skill index with {len(self.skill_names)} skills")

        # Save company index
        if "companies" in self.indexes and self.indexes["companies"] is not None:
            faiss.write_index(
                self.indexes["companies"],
                str(self.index_dir / "companies.index"),
            )

            with open(self.index_dir / "companies_centroids.pkl", "wb") as f:
                pickle.dump(self._company_centroid_map, f)

            with open(self.index_dir / "companies_names.pkl", "wb") as f:
                pickle.dump(self.company_names, f)

            # Save centroids matrix for efficient search
            if self._all_company_centroids is not None:
                np.save(
                    self.index_dir / "companies_centroids_matrix.npy",
                    self._all_company_centroids,
                )

            logger.info(f"Saved company index with {len(self.company_names)} companies")

        # Save metadata
        metadata = {
            "model_version": self.model_version,
            "nlist": self._nlist,
            "default_nprobe": self._default_nprobe,
        }
        with open(self.index_dir / "index_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        # Write model version as text file for easy inspection
        with open(self.index_dir / "model_version.txt", "w") as f:
            f.write(self.model_version)

        logger.info(f"Saved all indexes to {self.index_dir}")

    def load(self) -> bool:
        """
        Load indexes from disk.

        Returns:
            True if indexes were loaded successfully, False if not found

        Raises:
            IndexCompatibilityError: If model version doesn't match
        """
        import faiss

        if not self.exists():
            logger.warning(f"No indexes found in {self.index_dir}")
            return False

        # Load and check metadata
        metadata_path = self.index_dir / "index_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            saved_version = metadata.get("model_version", "unknown")
            if saved_version != self.model_version:
                raise IndexCompatibilityError(
                    f"Index model version '{saved_version}' doesn't match "
                    f"current version '{self.model_version}'"
                )

            self._nlist = metadata.get("nlist", 0)
            self._default_nprobe = metadata.get("default_nprobe", 10)

        # Load job index
        job_index_path = self.index_dir / "jobs.index"
        if job_index_path.exists():
            self.indexes["jobs"] = faiss.read_index(str(job_index_path))

            uuids_path = self.index_dir / "jobs_uuids.npy"
            if uuids_path.exists():
                uuids = np.load(uuids_path, allow_pickle=True)
                self.uuid_to_idx = {uuid: idx for idx, uuid in enumerate(uuids)}
                self.idx_to_uuid = {idx: uuid for idx, uuid in enumerate(uuids)}

            logger.info(f"Loaded job index with {self.indexes['jobs'].ntotal} vectors")

        # Load skill index
        skill_index_path = self.index_dir / "skills.index"
        if skill_index_path.exists():
            self.indexes["skills"] = faiss.read_index(str(skill_index_path))

            names_path = self.index_dir / "skills_names.pkl"
            if names_path.exists():
                with open(names_path, "rb") as f:
                    self.skill_names = pickle.load(f)
                self.skill_to_idx = {name: idx for idx, name in enumerate(self.skill_names)}

            logger.info(f"Loaded skill index with {len(self.skill_names)} skills")

        # Load company index
        company_index_path = self.index_dir / "companies.index"
        if company_index_path.exists():
            self.indexes["companies"] = faiss.read_index(str(company_index_path))

            centroids_path = self.index_dir / "companies_centroids.pkl"
            if centroids_path.exists():
                with open(centroids_path, "rb") as f:
                    self._company_centroid_map = pickle.load(f)

            names_path = self.index_dir / "companies_names.pkl"
            if names_path.exists():
                with open(names_path, "rb") as f:
                    self.company_names = pickle.load(f)

            centroids_matrix_path = self.index_dir / "companies_centroids_matrix.npy"
            if centroids_matrix_path.exists():
                self._all_company_centroids = np.load(centroids_matrix_path)

            logger.info(f"Loaded company index with {len(self.company_names)} companies")

        return True

    def exists(self) -> bool:
        """
        Check if saved indexes exist.

        Returns:
            True if at least the job index exists
        """
        return (self.index_dir / "jobs.index").exists()

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_stats(self) -> dict:
        """
        Get index statistics.

        Returns:
            Dict with size, memory usage, and configuration for each index type
        """
        stats = {
            "index_dir": str(self.index_dir),
            "model_version": self.model_version,
            "indexes": {},
        }

        # Job index stats
        if "jobs" in self.indexes and self.indexes["jobs"] is not None:
            job_index = self.indexes["jobs"]
            job_stats = {
                "total_vectors": job_index.ntotal,
                "dimension": self.DIMENSION,
                "nlist": self._nlist,
                "default_nprobe": self._default_nprobe,
                "index_type": type(job_index).__name__,
            }

            # Estimate memory usage (32-bit floats)
            job_stats["estimated_memory_mb"] = (
                job_index.ntotal * self.DIMENSION * 4 / (1024 * 1024)
            )

            stats["indexes"]["jobs"] = job_stats

        # Skill index stats
        if "skills" in self.indexes and self.indexes["skills"] is not None:
            stats["indexes"]["skills"] = {
                "total_skills": len(self.skill_names),
                "index_type": type(self.indexes["skills"]).__name__,
            }

        # Company index stats
        if "companies" in self.indexes and self.indexes["companies"] is not None:
            stats["indexes"]["companies"] = {
                "total_companies": len(self.company_names),
                "total_centroids": self.indexes["companies"].ntotal,
                "avg_centroids_per_company": (
                    self.indexes["companies"].ntotal / max(1, len(self.company_names))
                ),
                "index_type": type(self.indexes["companies"]).__name__,
            }

        return stats

    def is_compatible(self, model_version: str) -> bool:
        """
        Check if index is compatible with given model version.

        Embeddings from different models are not comparable, so the index
        must be rebuilt if the model changes.

        Args:
            model_version: Model version to check compatibility with

        Returns:
            True if compatible
        """
        return self.model_version == model_version

    def _calculate_nlist(self, n_vectors: int) -> int:
        """
        Calculate optimal nlist for IVFFlat.

        Rule of thumb: sqrt(n) clusters provides good balance.
        Clamped to [16, 4096] for practical performance.

        Args:
            n_vectors: Number of vectors to index

        Returns:
            Recommended nlist value
        """
        nlist = int(np.sqrt(n_vectors))
        nlist = max(nlist, 16)    # Minimum for small datasets
        nlist = min(nlist, 4096)  # Maximum practical value
        return nlist

    def _calculate_nprobe(self, nlist: int) -> int:
        """
        Calculate default nprobe for searches.

        Higher nprobe = better recall, slower search.
        Rule of thumb: nlist / 10 is a good starting point.

        Args:
            nlist: Number of clusters

        Returns:
            Recommended nprobe value
        """
        return max(1, nlist // 10)
