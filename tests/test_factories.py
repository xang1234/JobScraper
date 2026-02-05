"""
Tests for test data factories.

These tests ensure our factories generate valid data that can be
used reliably in other tests.
"""

import pytest
from datetime import date

from src.mcf.models import Job

from .factories import (
    generate_test_job,
    generate_test_jobs,
    generate_similar_jobs,
    generate_company_job_set,
    generate_salary_range_jobs,
    generate_salary,
    generate_company,
    generate_skills,
    generate_categories,
    generate_address,
    generate_metadata,
)


class TestGenerateTestJob:
    """Tests for the main job factory."""

    def test_generates_valid_job(self):
        """Test generated job has all required fields."""
        job = generate_test_job()

        assert isinstance(job, Job)
        assert job.uuid is not None
        assert job.title is not None
        assert len(job.title) > 0

    def test_job_has_nested_objects(self):
        """Test job has all nested objects populated."""
        job = generate_test_job()

        assert job.salary is not None
        assert job.postedCompany is not None
        assert len(job.skills) > 0
        assert len(job.categories) > 0
        assert job.address is not None
        assert len(job.employmentTypes) > 0
        assert len(job.positionLevels) > 0
        assert job.metadata is not None

    def test_computed_fields_work(self):
        """Test computed fields return expected values."""
        job = generate_test_job()

        # These should not raise errors
        assert job.company_name is not None
        assert job.salary_min is not None
        assert job.employment_type is not None
        assert job.skills_list is not None

    def test_title_override(self):
        """Test title can be overridden."""
        job = generate_test_job(title="Custom Title")
        assert job.title == "Custom Title"

    def test_company_override(self):
        """Test company name can be overridden."""
        job = generate_test_job(company_name="Custom Corp")
        assert job.company_name == "Custom Corp"

    def test_salary_override(self):
        """Test salary can be overridden."""
        job = generate_test_job(salary_min=12000, salary_max=15000)

        assert job.salary_min == 12000
        assert job.salary_max == 15000

    def test_skills_override(self):
        """Test skills can be overridden."""
        custom_skills = ["Rust", "Go", "Erlang"]
        job = generate_test_job(skills=custom_skills)

        skill_names = [s.skill for s in job.skills]
        assert set(skill_names) == set(custom_skills)

    def test_uuid_override(self):
        """Test UUID can be overridden."""
        job = generate_test_job(job_uuid="custom-uuid-12345")
        assert job.uuid == "custom-uuid-12345"

    def test_to_flat_dict_works(self):
        """Test generated job can be converted to flat dict."""
        job = generate_test_job()
        flat = job.to_flat_dict()

        assert isinstance(flat, dict)
        assert "uuid" in flat
        assert "title" in flat
        assert "company_name" in flat


class TestGenerateTestJobs:
    """Tests for batch job generation."""

    def test_generates_n_jobs(self):
        """Test correct number of jobs generated."""
        jobs = generate_test_jobs(10)
        assert len(jobs) == 10

    def test_jobs_have_unique_uuids(self):
        """Test all generated jobs have unique UUIDs."""
        jobs = generate_test_jobs(50)
        uuids = [j.uuid for j in jobs]

        assert len(uuids) == len(set(uuids))

    def test_jobs_have_variety(self):
        """Test jobs have varied titles and companies."""
        jobs = generate_test_jobs(20)

        titles = {j.title for j in jobs}
        companies = {j.company_name for j in jobs}

        # Should have some variety
        assert len(titles) > 1
        assert len(companies) > 1


class TestGenerateSimilarJobs:
    """Tests for similar job generation."""

    def test_generates_title_variations(self):
        """Test jobs have title variations."""
        jobs = generate_similar_jobs("Data Scientist", n=5)

        titles = [j.title for j in jobs]

        assert len(jobs) == 5
        # All titles should contain "Data Scientist"
        assert all("Data Scientist" in t for t in titles)
        # But titles should vary
        assert len(set(titles)) > 1

    def test_similar_jobs_share_skills(self):
        """Test similar jobs have overlapping skills."""
        jobs = generate_similar_jobs("Software Engineer", n=3)

        skill_sets = [set(s.skill for s in j.skills) for j in jobs]

        # There should be some skill overlap
        common = skill_sets[0] & skill_sets[1]
        assert len(common) > 0


class TestGenerateCompanyJobSet:
    """Tests for company-grouped job generation."""

    def test_all_jobs_same_company(self):
        """Test all jobs belong to same company."""
        jobs = generate_company_job_set("Test Corp Pte Ltd", n=10)

        companies = {j.company_name for j in jobs}

        assert len(companies) == 1
        assert "Test Corp Pte Ltd" in companies

    def test_generates_correct_count(self):
        """Test correct number of jobs generated."""
        jobs = generate_company_job_set("Test Corp", n=7)
        assert len(jobs) == 7


class TestGenerateSalaryRangeJobs:
    """Tests for salary-ranged job generation."""

    def test_salaries_in_range(self):
        """Test all salaries fall within specified range."""
        jobs = generate_salary_range_jobs(
            min_salary=8000,
            max_salary=12000,
            n=10,
        )

        for job in jobs:
            assert job.salary_min >= 8000
            assert job.salary_max <= 12000
            assert job.salary_min <= job.salary_max


class TestHelperFactories:
    """Tests for individual component factories."""

    def test_generate_salary(self):
        """Test salary generation."""
        salary = generate_salary(min_salary=5000, max_salary=8000)

        assert salary.minimum == 5000
        assert salary.maximum == 8000
        assert salary.type is not None

    def test_generate_company(self):
        """Test company generation."""
        company = generate_company(name="Custom Inc")

        assert company.name == "Custom Inc"
        assert company.description is not None

    def test_generate_skills(self):
        """Test skill generation."""
        skills = generate_skills(n=5)

        assert len(skills) == 5
        assert all(s.skill is not None for s in skills)

    def test_generate_categories(self):
        """Test category generation."""
        categories = generate_categories(n=3)

        assert len(categories) <= 3  # May be limited by pool size
        assert all(c.category is not None for c in categories)

    def test_generate_address(self):
        """Test address generation."""
        address = generate_address()

        assert address.district is not None
        assert address.region is not None
        assert address.postalCode is not None

    def test_generate_metadata(self):
        """Test metadata generation."""
        metadata = generate_metadata(posted_days_ago=5, applications=50)

        assert metadata.totalNumberJobApplication == 50

        # Verify posting date is approximately 5 days ago
        posting_date = date.fromisoformat(metadata.newPostingDate)
        days_diff = (date.today() - posting_date).days
        assert days_diff == 5
