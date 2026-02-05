"""
Tests for Pydantic models.

These tests verify that models correctly parse API responses,
compute derived fields, and handle edge cases.
"""

import pytest
from datetime import date

from src.mcf.models import (
    Job,
    Salary,
    SalaryType,
    Company,
    Skill,
    Category,
    Address,
    EmploymentType,
    PositionLevel,
    JobMetadata,
)


class TestSalary:
    """Tests for Salary model."""

    def test_salary_with_all_fields(self):
        """Test salary with complete data."""
        salary = Salary(
            minimum=5000,
            maximum=8000,
            type=SalaryType(salaryType="Monthly"),
        )

        assert salary.minimum == 5000
        assert salary.maximum == 8000
        assert salary.salary_type == "Monthly"

    def test_salary_with_none_values(self):
        """Test salary with missing values."""
        salary = Salary()

        assert salary.minimum is None
        assert salary.maximum is None
        assert salary.salary_type == "Unknown"

    def test_salary_type_extraction(self):
        """Test salary type property."""
        salary = Salary(type=SalaryType(salaryType="Yearly"))
        assert salary.salary_type == "Yearly"

        salary_no_type = Salary(type=None)
        assert salary_no_type.salary_type == "Unknown"


class TestCompany:
    """Tests for Company model."""

    def test_company_with_all_fields(self):
        """Test company with complete data."""
        company = Company(
            name="Test Corp",
            uen="T12345678A",
            description="A test company",
        )

        assert company.name == "Test Corp"
        assert company.uen == "T12345678A"

    def test_company_with_empty_name(self):
        """Test company with default name."""
        company = Company()
        assert company.name == ""


class TestAddress:
    """Tests for Address model."""

    def test_formatted_address(self):
        """Test address formatting."""
        address = Address(
            block="10",
            street="Marina Boulevard",
            floor="05",
            unit="01",
            postalCode="018983",
        )

        formatted = address.formatted
        assert "10" in formatted
        assert "Marina Boulevard" in formatted
        assert "#05-01" in formatted
        assert "S(018983)" in formatted

    def test_empty_address_formatting(self):
        """Test empty address returns empty string."""
        address = Address()
        assert address.formatted == ""

    def test_partial_address_formatting(self):
        """Test partial address formatting."""
        address = Address(street="Orchard Road", postalCode="238823")
        formatted = address.formatted
        assert "Orchard Road" in formatted
        assert "S(238823)" in formatted


class TestJob:
    """Tests for Job model."""

    @pytest.fixture
    def complete_job(self) -> Job:
        """Create a complete job for testing."""
        return Job(
            uuid="test-uuid-12345",
            title="Senior Data Scientist",
            description="<p>We need a data scientist.</p>",
            salary=Salary(
                minimum=10000,
                maximum=15000,
                type=SalaryType(salaryType="Monthly"),
            ),
            postedCompany=Company(name="Test Corp", uen="T12345678A"),
            skills=[
                Skill(skill="Python", isKeySkill=True),
                Skill(skill="Machine Learning", isKeySkill=True),
                Skill(skill="SQL", isKeySkill=False),
            ],
            categories=[
                Category(category="Technology", id=1),
                Category(category="Data Science", id=2),
            ],
            address=Address(
                district="Downtown Core",
                region="Central",
            ),
            employmentTypes=[EmploymentType(employmentType="Full Time")],
            positionLevels=[PositionLevel(position="Senior")],
            minimumYearsExperience=5,
            metadata=JobMetadata(
                totalNumberJobApplication=42,
                newPostingDate="2024-01-15",
                expiryDate="2024-02-15",
            ),
        )

    def test_computed_company_fields(self, complete_job: Job):
        """Test company_name and company_uen computed fields."""
        assert complete_job.company_name == "Test Corp"
        assert complete_job.company_uen == "T12345678A"

    def test_computed_salary_fields(self, complete_job: Job):
        """Test salary computed fields."""
        assert complete_job.salary_min == 10000
        assert complete_job.salary_max == 15000
        assert complete_job.salary_type == "Monthly"

    def test_skills_list(self, complete_job: Job):
        """Test skills_list computed field."""
        assert complete_job.skills_list == "Python, Machine Learning, SQL"

    def test_categories_list(self, complete_job: Job):
        """Test categories_list computed field."""
        assert complete_job.categories_list == "Technology, Data Science"

    def test_employment_type(self, complete_job: Job):
        """Test employment_type computed field."""
        assert complete_job.employment_type == "Full Time"

    def test_seniority(self, complete_job: Job):
        """Test seniority computed field."""
        assert complete_job.seniority == "Senior"

    def test_location_fields(self, complete_job: Job):
        """Test location computed fields."""
        assert complete_job.district == "Downtown Core"
        assert complete_job.region == "Central"

    def test_applications_count(self, complete_job: Job):
        """Test applications_count computed field."""
        assert complete_job.applications_count == 42

    def test_posted_date_parsing(self, complete_job: Job):
        """Test posted_date parsing."""
        assert complete_job.posted_date == date(2024, 1, 15)

    def test_expiry_date_parsing(self, complete_job: Job):
        """Test expiry_date parsing."""
        assert complete_job.expiry_date == date(2024, 2, 15)

    def test_description_text_html_stripping(self, complete_job: Job):
        """Test HTML is stripped from description."""
        assert complete_job.description_text == "We need a data scientist."

    def test_job_url_generation(self, complete_job: Job):
        """Test job URL generation."""
        url = complete_job.job_url
        assert "mycareersfuture.gov.sg/job/" in url
        assert "senior-data-scientist" in url
        assert complete_job.uuid in url

    def test_to_flat_dict(self, complete_job: Job):
        """Test conversion to flat dictionary."""
        flat = complete_job.to_flat_dict()

        assert flat["uuid"] == "test-uuid-12345"
        assert flat["title"] == "Senior Data Scientist"
        assert flat["company_name"] == "Test Corp"
        assert flat["salary_min"] == 10000
        assert flat["salary_max"] == 15000
        assert flat["employment_type"] == "Full Time"
        assert flat["seniority"] == "Senior"
        assert "Python" in flat["skills"]

    def test_job_with_missing_nested_objects(self):
        """Test job handles missing nested objects gracefully."""
        minimal_job = Job(
            uuid="minimal-uuid",
            title="Minimal Job",
        )

        assert minimal_job.company_name == ""
        assert minimal_job.company_uen is None
        assert minimal_job.salary_min is None
        assert minimal_job.salary_max is None
        assert minimal_job.salary_type == "Unknown"
        assert minimal_job.employment_type == "Unknown"
        assert minimal_job.seniority == "Unknown"
        assert minimal_job.location == ""
        assert minimal_job.applications_count == 0

    def test_job_with_empty_lists(self):
        """Test job handles empty skill/category lists."""
        job = Job(
            uuid="empty-lists-uuid",
            title="Empty Lists Job",
            skills=[],
            categories=[],
        )

        assert job.skills_list == ""
        assert job.categories_list == ""


class TestJobMetadata:
    """Tests for JobMetadata model."""

    def test_metadata_defaults(self):
        """Test default values for metadata."""
        metadata = JobMetadata()

        assert metadata.totalNumberJobApplication == 0
        assert metadata.expiryDate is None
        assert metadata.newPostingDate is None
        assert metadata.isPostedOnBehalf is False

    def test_metadata_with_iso_dates(self):
        """Test metadata with ISO format dates."""
        metadata = JobMetadata(
            newPostingDate="2024-01-15T00:00:00Z",
            expiryDate="2024-02-15T00:00:00Z",
        )

        assert metadata.newPostingDate == "2024-01-15T00:00:00Z"
        assert metadata.expiryDate == "2024-02-15T00:00:00Z"


class TestSkill:
    """Tests for Skill model."""

    def test_skill_basic(self):
        """Test basic skill creation."""
        skill = Skill(skill="Python")
        assert skill.skill == "Python"
        assert skill.isKeySkill is False

    def test_key_skill(self):
        """Test key skill flag."""
        skill = Skill(skill="Machine Learning", isKeySkill=True)
        assert skill.isKeySkill is True


class TestCategory:
    """Tests for Category model."""

    def test_category_basic(self):
        """Test basic category creation."""
        category = Category(category="Technology")
        assert category.category == "Technology"
        assert category.id is None

    def test_category_with_id(self):
        """Test category with ID."""
        category = Category(category="Data Science", id=42)
        assert category.id == 42
