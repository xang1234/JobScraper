"""
Query expansion using skill clusters for synonym matching.

Expands user search queries with related terms from precomputed skill clusters.
This improves recall by matching different vocabulary:
- User: "ML engineer" → Expansion includes "Machine Learning", "Deep Learning"
- User: "DS" → Expansion includes "Data Science", "Data Scientist"
- User: "devops" → Expansion includes "Kubernetes", "Docker", "CI/CD"

The skill clusters are generated during `embed-generate` (Phase 1.3) using
agglomerative clustering on skill embeddings. Skills in the same cluster
are semantically related and make good expansion candidates.

Example:
    expander = QueryExpander.load(Path("data/embeddings"))
    expanded = expander.expand("ML engineer")
    # Returns: ["ML", "Machine Learning", "Deep Learning", "engineer"]
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expands search queries with related terms using skill clusters.

    Uses precomputed skill clusters (from embed-generate) to find synonyms
    and related terms. Supports multiple matching strategies:
    - Exact match (case-insensitive)
    - Prefix match ("python" matches "Python Programming")
    - Acronym match ("ML" matches "Machine Learning")
    - Substring match for compound terms

    Example:
        expander = QueryExpander.load(Path("data/embeddings"))

        # Basic expansion
        expanded = expander.expand("ML engineer")
        # ["ML", "Machine Learning", "Deep Learning", "engineer"]

        # Get related skills directly
        related = expander.get_related_skills("Python", k=5)
        # ["Pandas", "NumPy", "Django", ...]
    """

    def __init__(
        self,
        skill_clusters: dict[int, list[str]],
        skill_to_cluster: dict[str, int],
    ):
        """
        Initialize QueryExpander with precomputed cluster data.

        Args:
            skill_clusters: Mapping of cluster_id -> list of skill names
            skill_to_cluster: Mapping of skill_name -> cluster_id
        """
        self.skill_clusters = skill_clusters
        self.skill_to_cluster = skill_to_cluster

        # Build reverse lookup for fast case-insensitive matching
        self._skill_lower_map: dict[str, str] = {
            s.lower(): s for s in skill_to_cluster.keys()
        }

        # Precompute acronyms for quick lookup
        self._acronym_map: dict[str, str] = {}
        for skill in skill_to_cluster.keys():
            acronym = self._compute_acronym(skill)
            if acronym and len(acronym) >= 2:
                self._acronym_map[acronym] = skill

        logger.debug(
            f"Initialized QueryExpander with {len(skill_to_cluster)} skills "
            f"in {len(skill_clusters)} clusters"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "QueryExpander":
        """
        Load precomputed clusters from disk.

        Files are created by embed-generate (Phase 1.3):
        - skill_clusters.pkl: Dict[int, list[str]]
        - skill_to_cluster.pkl: Dict[str, int]

        Args:
            index_dir: Directory containing cluster pickle files

        Returns:
            Initialized QueryExpander

        Raises:
            FileNotFoundError: If cluster files don't exist
        """
        index_dir = Path(index_dir)
        clusters_path = index_dir / "skill_clusters.pkl"
        mapping_path = index_dir / "skill_to_cluster.pkl"

        if not clusters_path.exists():
            raise FileNotFoundError(
                f"Skill clusters not found at {clusters_path}. "
                f"Run 'mcf embed-generate' first."
            )

        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Skill-to-cluster mapping not found at {mapping_path}. "
                f"Run 'mcf embed-generate' first."
            )

        with open(clusters_path, "rb") as f:
            skill_clusters = pickle.load(f)

        with open(mapping_path, "rb") as f:
            skill_to_cluster = pickle.load(f)

        logger.info(
            f"Loaded skill clusters: {len(skill_to_cluster)} skills "
            f"in {len(skill_clusters)} clusters"
        )

        return cls(skill_clusters, skill_to_cluster)

    def expand(self, query: str, max_expansions: int = 3) -> list[str]:
        """
        Expand query with related terms from skill clusters.

        Each word in the query is checked against known skills. If a match
        is found, related skills from the same cluster are added as expansions.

        Args:
            query: Original search query (e.g., "ML engineer")
            max_expansions: Max additional terms to add per matched word

        Returns:
            List of terms: original words + expansions, deduplicated
        """
        if not query or not query.strip():
            return []

        # Tokenize query into words
        words = self._tokenize(query)
        expanded: list[str] = list(words)  # Start with original words

        for word in words:
            # Try to find a matching skill
            matching_skill = self._find_matching_skill(word)

            if matching_skill:
                # Get related skills from same cluster
                related = self.get_related_skills(matching_skill, k=max_expansions)
                expanded.extend(related)

                logger.debug(
                    f"Expanded '{word}' via '{matching_skill}' -> {related}"
                )

        # Deduplicate while preserving order
        result = self._deduplicate(expanded)

        if len(result) > len(words):
            logger.info(
                f"Query expansion: '{query}' -> {len(result)} terms "
                f"(+{len(result) - len(words)} expansions)"
            )

        return result

    def get_related_skills(self, skill: str, k: int = 5) -> list[str]:
        """
        Get skills related to a given skill (from same cluster).

        Args:
            skill: Skill name to find relatives for
            k: Maximum number of related skills to return

        Returns:
            List of related skill names (excluding input skill)
        """
        cluster_id = self.skill_to_cluster.get(skill)
        if cluster_id is None:
            return []

        cluster_skills = self.skill_clusters.get(cluster_id, [])

        # Return other skills from same cluster (not the input skill)
        related = [s for s in cluster_skills if s != skill]
        return related[:k]

    def get_cluster_for_skill(self, skill: str) -> Optional[int]:
        """
        Get cluster ID for a skill.

        Args:
            skill: Skill name

        Returns:
            Cluster ID or None if skill not found
        """
        return self.skill_to_cluster.get(skill)

    def get_all_skills_in_cluster(self, cluster_id: int) -> list[str]:
        """
        Get all skills in a cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            List of skill names in the cluster
        """
        return self.skill_clusters.get(cluster_id, [])

    def _find_matching_skill(self, word: str) -> Optional[str]:
        """
        Find skill that matches the word using multiple strategies.

        Matching strategies (in order of preference):
        1. Exact match (case-insensitive): "python" -> "Python"
        2. Acronym match: "ML" -> "Machine Learning"
        3. Prefix match (min 3 chars): "pyth" -> "Python"
        4. Word boundary match: "learning" -> "Machine Learning"

        Args:
            word: Word to match against skills

        Returns:
            Matching skill name or None
        """
        word_lower = word.lower()
        word_upper = word.upper()

        # Strategy 1: Exact match (case-insensitive)
        if word_lower in self._skill_lower_map:
            return self._skill_lower_map[word_lower]

        # Strategy 2: Acronym match (for 2-4 letter uppercase or all-caps words)
        if len(word) <= 4 and (word.isupper() or word == word_upper):
            if word_upper in self._acronym_map:
                return self._acronym_map[word_upper]

        # Strategy 3: Prefix match (min 3 chars to avoid false positives)
        if len(word_lower) >= 3:
            for skill_lower, skill in self._skill_lower_map.items():
                if skill_lower.startswith(word_lower):
                    return skill

        # Strategy 4: Word boundary match for multi-word skills
        if len(word_lower) >= 4:
            for skill_lower, skill in self._skill_lower_map.items():
                # Check if word appears as a complete word in skill
                if re.search(rf'\b{re.escape(word_lower)}\b', skill_lower):
                    return skill

        return None

    def _tokenize(self, query: str) -> list[str]:
        """
        Tokenize query into words.

        Handles:
        - Whitespace separation
        - Preserves acronyms (uppercase sequences)
        - Removes punctuation

        Args:
            query: Raw query string

        Returns:
            List of word tokens
        """
        # Replace punctuation with spaces (except hyphens in compound words)
        cleaned = re.sub(r'[^\w\s-]', ' ', query)
        # Split on whitespace
        words = cleaned.split()
        # Filter empty strings
        return [w for w in words if w]

    def _deduplicate(self, terms: list[str]) -> list[str]:
        """
        Deduplicate terms while preserving order.

        Case-insensitive deduplication (keeps first occurrence).

        Args:
            terms: List of terms to deduplicate

        Returns:
            Deduplicated list
        """
        seen: set[str] = set()
        result: list[str] = []

        for term in terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                result.append(term)

        return result

    @staticmethod
    def _compute_acronym(skill: str) -> str:
        """
        Compute acronym for a multi-word skill.

        Examples:
            "Machine Learning" -> "ML"
            "Natural Language Processing" -> "NLP"
            "Python" -> "P" (single word, won't be used)

        Args:
            skill: Skill name

        Returns:
            Uppercase acronym
        """
        words = skill.split()
        if len(words) < 2:
            return ""
        return ''.join(w[0] for w in words if w).upper()

    def get_stats(self) -> dict:
        """
        Get statistics about the expander.

        Returns:
            Dict with skill count, cluster count, and average cluster size
        """
        cluster_sizes = [len(skills) for skills in self.skill_clusters.values()]

        return {
            "total_skills": len(self.skill_to_cluster),
            "total_clusters": len(self.skill_clusters),
            "avg_cluster_size": (
                sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
            ),
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "acronyms_indexed": len(self._acronym_map),
        }
