"""
Tests for QueryExpander.

Tests cover:
- Loading from disk
- Query expansion with various matching strategies
- Related skills lookup
- Acronym matching
- Edge cases and error handling
"""

import pickle
import tempfile
from pathlib import Path

import pytest

from src.mcf.embeddings import QueryExpander


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_index_dir():
    """Provide a temporary directory for cluster files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_clusters():
    """Sample skill clusters for testing."""
    skill_clusters = {
        0: ["Python", "Pandas", "NumPy", "Scikit-learn"],
        1: ["Machine Learning", "Deep Learning", "Neural Networks", "TensorFlow"],
        2: ["AWS", "Azure", "Google Cloud Platform", "Cloud Computing"],
        3: ["Docker", "Kubernetes", "CI/CD", "DevOps"],
        4: ["Data Science", "Data Analysis", "Statistics", "Data Visualization"],
    }

    skill_to_cluster = {}
    for cluster_id, skills in skill_clusters.items():
        for skill in skills:
            skill_to_cluster[skill] = cluster_id

    return skill_clusters, skill_to_cluster


@pytest.fixture
def expander(sample_clusters):
    """Create QueryExpander with sample clusters."""
    skill_clusters, skill_to_cluster = sample_clusters
    return QueryExpander(skill_clusters, skill_to_cluster)


@pytest.fixture
def saved_clusters(temp_index_dir, sample_clusters):
    """Save sample clusters to disk and return path."""
    skill_clusters, skill_to_cluster = sample_clusters

    with open(temp_index_dir / "skill_clusters.pkl", "wb") as f:
        pickle.dump(skill_clusters, f)

    with open(temp_index_dir / "skill_to_cluster.pkl", "wb") as f:
        pickle.dump(skill_to_cluster, f)

    return temp_index_dir


# =============================================================================
# Load Tests
# =============================================================================


class TestLoad:
    """Tests for QueryExpander.load() method."""

    def test_load_from_disk(self, saved_clusters):
        """Test loading clusters from pickle files."""
        expander = QueryExpander.load(saved_clusters)

        assert len(expander.skill_to_cluster) == 20  # 5 clusters × 4 skills
        assert len(expander.skill_clusters) == 5

    def test_load_missing_clusters(self, temp_index_dir):
        """Test error when clusters file missing."""
        with pytest.raises(FileNotFoundError, match="skill_clusters"):
            QueryExpander.load(temp_index_dir)

    def test_load_missing_mapping(self, temp_index_dir):
        """Test error when mapping file missing."""
        # Create only clusters file
        with open(temp_index_dir / "skill_clusters.pkl", "wb") as f:
            pickle.dump({}, f)

        with pytest.raises(FileNotFoundError, match="skill_to_cluster"):
            QueryExpander.load(temp_index_dir)


# =============================================================================
# Expand Tests
# =============================================================================


class TestExpand:
    """Tests for QueryExpander.expand() method."""

    def test_expand_exact_match(self, expander):
        """Test expansion with exact skill match."""
        expanded = expander.expand("Python developer")

        assert "Python" in expanded or "python" in expanded
        # Should have expansions from Python cluster
        assert any(s in expanded for s in ["Pandas", "NumPy", "Scikit-learn"])

    def test_expand_case_insensitive(self, expander):
        """Test case-insensitive matching."""
        expanded = expander.expand("python DEVELOPER")

        assert "python" in expanded
        # Should still find Python cluster
        assert any(s in expanded for s in ["Pandas", "NumPy", "Scikit-learn"])

    def test_expand_acronym_ml(self, expander):
        """Test acronym expansion for ML."""
        expanded = expander.expand("ML engineer")

        # "ML" should match "Machine Learning"
        # So we should see related skills from cluster 1
        assert any(
            s in expanded
            for s in ["Machine Learning", "Deep Learning", "Neural Networks", "TensorFlow"]
        )

    def test_expand_acronym_ds(self, expander):
        """Test acronym expansion for DS (Data Science)."""
        expanded = expander.expand("DS analyst")

        # "DS" should match "Data Science"
        assert any(
            s in expanded
            for s in ["Data Science", "Data Analysis", "Statistics"]
        )

    def test_expand_preserves_original(self, expander):
        """Test that original words are preserved."""
        expanded = expander.expand("Python developer")

        # Both original words should be present
        assert "Python" in expanded or "python" in expanded
        assert "developer" in expanded

    def test_expand_no_match(self, expander):
        """Test expansion when no skills match."""
        expanded = expander.expand("accounting manager")

        # Should just return original words
        assert expanded == ["accounting", "manager"]

    def test_expand_empty_query(self, expander):
        """Test expansion with empty query."""
        assert expander.expand("") == []
        assert expander.expand("   ") == []

    def test_expand_max_expansions(self, expander):
        """Test max_expansions parameter limits results."""
        expanded_1 = expander.expand("Python", max_expansions=1)
        expanded_3 = expander.expand("Python", max_expansions=3)

        # More expansions should mean more terms
        assert len(expanded_3) >= len(expanded_1)

    def test_expand_deduplicates(self, expander):
        """Test that expansion deduplicates results."""
        expanded = expander.expand("Python python PYTHON")

        # Should only have one "python" variant
        python_count = sum(1 for t in expanded if t.lower() == "python")
        assert python_count == 1

    def test_expand_multiple_skills(self, expander):
        """Test expansion with multiple matchable skills."""
        expanded = expander.expand("Python AWS engineer")

        # Should have expansions from both clusters
        has_python_related = any(s in expanded for s in ["Pandas", "NumPy"])
        has_aws_related = any(s in expanded for s in ["Azure", "Google Cloud Platform"])

        assert has_python_related
        assert has_aws_related


# =============================================================================
# Related Skills Tests
# =============================================================================


class TestRelatedSkills:
    """Tests for get_related_skills method."""

    def test_get_related_skills_basic(self, expander):
        """Test getting related skills."""
        related = expander.get_related_skills("Python", k=5)

        assert len(related) <= 5
        assert "Python" not in related  # Should not include input
        assert all(s in ["Pandas", "NumPy", "Scikit-learn"] for s in related)

    def test_get_related_skills_limit(self, expander):
        """Test k parameter limits results."""
        related_1 = expander.get_related_skills("Python", k=1)
        related_3 = expander.get_related_skills("Python", k=3)

        assert len(related_1) == 1
        assert len(related_3) == 3

    def test_get_related_skills_unknown(self, expander):
        """Test with unknown skill."""
        related = expander.get_related_skills("UnknownSkill123")

        assert related == []


# =============================================================================
# Matching Strategy Tests
# =============================================================================


class TestMatchingStrategies:
    """Tests for different matching strategies."""

    def test_prefix_match(self, expander):
        """Test prefix matching for partial words."""
        # "pyth" should match "Python"
        match = expander._find_matching_skill("pyth")
        assert match == "Python"

    def test_prefix_match_min_length(self, expander):
        """Test that prefix matching requires min 3 chars."""
        # "py" too short for prefix match
        match = expander._find_matching_skill("py")
        assert match is None

    def test_word_boundary_match(self, expander):
        """Test word boundary matching in multi-word skills."""
        # "learning" should match "Machine Learning" or "Deep Learning"
        match = expander._find_matching_skill("learning")
        assert match in ["Machine Learning", "Deep Learning"]

    def test_acronym_computation(self, expander):
        """Test acronym computation."""
        assert expander._compute_acronym("Machine Learning") == "ML"
        assert expander._compute_acronym("Natural Language Processing") == "NLP"
        assert expander._compute_acronym("Python") == ""  # Single word


# =============================================================================
# Utility Tests
# =============================================================================


class TestUtilities:
    """Tests for utility methods."""

    def test_get_cluster_for_skill(self, expander):
        """Test getting cluster ID for skill."""
        cluster_id = expander.get_cluster_for_skill("Python")
        assert cluster_id == 0

        cluster_id = expander.get_cluster_for_skill("Unknown")
        assert cluster_id is None

    def test_get_all_skills_in_cluster(self, expander):
        """Test getting all skills in a cluster."""
        skills = expander.get_all_skills_in_cluster(0)
        assert set(skills) == {"Python", "Pandas", "NumPy", "Scikit-learn"}

        skills = expander.get_all_skills_in_cluster(999)
        assert skills == []

    def test_get_stats(self, expander):
        """Test getting expander statistics."""
        stats = expander.get_stats()

        assert stats["total_skills"] == 20
        assert stats["total_clusters"] == 5
        assert stats["avg_cluster_size"] == 4.0
        assert stats["max_cluster_size"] == 4
        assert stats["acronyms_indexed"] > 0

    def test_tokenize(self, expander):
        """Test query tokenization."""
        tokens = expander._tokenize("Python, AWS & Docker")
        assert "Python" in tokens
        assert "AWS" in tokens
        assert "Docker" in tokens
        assert "," not in tokens
        assert "&" not in tokens

    def test_tokenize_preserves_hyphens(self, expander):
        """Test that hyphens in compound words are preserved."""
        tokens = expander._tokenize("CI-CD pipeline")
        # Hyphen preserved within word
        assert "CI-CD" in tokens

    def test_deduplicate_preserves_order(self, expander):
        """Test deduplication preserves first occurrence."""
        result = expander._deduplicate(["a", "B", "a", "c", "b"])
        assert result == ["a", "B", "c"]
