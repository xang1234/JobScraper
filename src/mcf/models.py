"""
Pydantic models for MyCareersFuture API responses.

These models provide type validation and automatic conversion for the JSON
data returned by the MCF API, ensuring data integrity and enabling IDE support.
"""

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator, computed_field
import re


class SalaryType(BaseModel):
    """Salary type information (Monthly, Yearly, Hourly, etc.)"""
    salaryType: Optional[str] = None


class Salary(BaseModel):
    """Salary range information with type."""
    minimum: Optional[int] = Field(default=None, alias="minimum")
    maximum: Optional[int] = Field(default=None, alias="maximum")
    type: Optional[SalaryType] = None

    @property
    def salary_type(self) -> str:
        """Get the salary type as a string."""
        if self.type and self.type.salaryType:
            return self.type.salaryType
        return "Unknown"


class Company(BaseModel):
    """Company information from the job posting."""
    name: str = ""
    uen: Optional[str] = None
    description: Optional[str] = None
    logoUri: Optional[str] = None


class Skill(BaseModel):
    """Skill requirement for a job."""
    skill: str
    isKeySkill: Optional[bool] = False


class Category(BaseModel):
    """Job category/classification."""
    category: str
    id: Optional[int] = None


class Address(BaseModel):
    """Location/address information."""
    block: Optional[str] = None
    street: Optional[str] = None
    floor: Optional[str] = None
    unit: Optional[str] = None
    postalCode: Optional[str] = None
    district: Optional[str] = None
    region: Optional[str] = None

    @property
    def formatted(self) -> str:
        """Return formatted address string."""
        parts = []
        if self.block:
            parts.append(self.block)
        if self.street:
            parts.append(self.street)
        if self.floor and self.unit:
            parts.append(f"#{self.floor}-{self.unit}")
        if self.postalCode:
            parts.append(f"S({self.postalCode})")
        return " ".join(parts) if parts else ""


class EmploymentType(BaseModel):
    """Employment type information."""
    employmentType: str = "Unknown"


class PositionLevel(BaseModel):
    """Position level/seniority information."""
    position: str = "Unknown"


class JobMetadata(BaseModel):
    """Metadata about the job posting."""
    totalNumberJobApplication: int = 0
    expiryDate: Optional[str] = None
    newPostingDate: Optional[str] = None
    originalPostingDate: Optional[str] = None
    isPostedOnBehalf: bool = False


class Job(BaseModel):
    """
    Complete job listing from MyCareersFuture.

    This model maps the API response to a structured object with
    proper types and computed fields for easy data access.
    """
    uuid: str
    title: str
    description: str = ""
    salary: Optional[Salary] = None
    postedCompany: Optional[Company] = None
    skills: list[Skill] = Field(default_factory=list)
    categories: list[Category] = Field(default_factory=list)
    address: Optional[Address] = None
    employmentTypes: list[EmploymentType] = Field(default_factory=list)
    positionLevels: list[PositionLevel] = Field(default_factory=list)
    minimumYearsExperience: Optional[int] = None
    metadata: Optional[JobMetadata] = None

    # Additional fields that may be present
    numberOfVacancies: int = 1
    ssocCode: Optional[str] = None

    @computed_field
    @property
    def company_name(self) -> str:
        """Get company name."""
        return self.postedCompany.name if self.postedCompany else ""

    @computed_field
    @property
    def company_uen(self) -> Optional[str]:
        """Get company UEN (unique entity number)."""
        return self.postedCompany.uen if self.postedCompany else None

    @computed_field
    @property
    def salary_min(self) -> Optional[int]:
        """Get minimum salary."""
        return self.salary.minimum if self.salary else None

    @computed_field
    @property
    def salary_max(self) -> Optional[int]:
        """Get maximum salary."""
        return self.salary.maximum if self.salary else None

    @computed_field
    @property
    def salary_type(self) -> str:
        """Get salary type (Monthly, Yearly, etc.)."""
        return self.salary.salary_type if self.salary else "Unknown"

    @computed_field
    @property
    def employment_type(self) -> str:
        """Get primary employment type."""
        if self.employmentTypes:
            return self.employmentTypes[0].employmentType
        return "Unknown"

    @computed_field
    @property
    def seniority(self) -> str:
        """Get position seniority level."""
        if self.positionLevels:
            return self.positionLevels[0].position
        return "Unknown"

    @computed_field
    @property
    def skills_list(self) -> str:
        """Get comma-separated list of skills."""
        return ", ".join(s.skill for s in self.skills)

    @computed_field
    @property
    def categories_list(self) -> str:
        """Get comma-separated list of categories."""
        return ", ".join(c.category for c in self.categories)

    @computed_field
    @property
    def location(self) -> str:
        """Get formatted location string."""
        return self.address.formatted if self.address else ""

    @computed_field
    @property
    def district(self) -> Optional[str]:
        """Get district name."""
        return self.address.district if self.address else None

    @computed_field
    @property
    def region(self) -> Optional[str]:
        """Get region name."""
        return self.address.region if self.address else None

    @computed_field
    @property
    def posted_date(self) -> Optional[date]:
        """Get the posting date."""
        if self.metadata and self.metadata.newPostingDate:
            try:
                return datetime.fromisoformat(
                    self.metadata.newPostingDate.replace("Z", "+00:00")
                ).date()
            except ValueError:
                return None
        return None

    @computed_field
    @property
    def expiry_date(self) -> Optional[date]:
        """Get the expiry date."""
        if self.metadata and self.metadata.expiryDate:
            try:
                return datetime.fromisoformat(
                    self.metadata.expiryDate.replace("Z", "+00:00")
                ).date()
            except ValueError:
                return None
        return None

    @computed_field
    @property
    def applications_count(self) -> int:
        """Get number of applications."""
        return self.metadata.totalNumberJobApplication if self.metadata else 0

    @computed_field
    @property
    def job_url(self) -> str:
        """Construct the URL to view this job on MCF."""
        # URL format: https://www.mycareersfuture.gov.sg/job/{slug}-{uuid}
        slug = re.sub(r'[^a-z0-9]+', '-', self.title.lower()).strip('-')
        return f"https://www.mycareersfuture.gov.sg/job/{slug}-{self.uuid}"

    @computed_field
    @property
    def description_text(self) -> str:
        """Get description with HTML tags stripped."""
        # Simple HTML tag removal - handles most cases
        text = re.sub(r'<[^>]+>', ' ', self.description)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def to_flat_dict(self) -> dict:
        """
        Convert to a flat dictionary suitable for CSV export.

        This produces the clean schema defined in the plan.
        """
        return {
            "uuid": self.uuid,
            "title": self.title,
            "company_name": self.company_name,
            "company_uen": self.company_uen,
            "description": self.description_text,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "salary_type": self.salary_type,
            "employment_type": self.employment_type,
            "seniority": self.seniority,
            "min_experience_years": self.minimumYearsExperience,
            "skills": self.skills_list,
            "categories": self.categories_list,
            "location": self.location,
            "district": self.district,
            "region": self.region,
            "posted_date": self.posted_date.isoformat() if self.posted_date else None,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "applications_count": self.applications_count,
            "job_url": self.job_url,
        }


class JobSearchResponse(BaseModel):
    """Response from the MCF job search API endpoint."""
    results: list[Job] = Field(default_factory=list)
    total: int = 0

    # Pagination info
    offset: int = 0
    limit: int = 20


class Checkpoint(BaseModel):
    """Checkpoint for resumable scraping."""
    search_query: str
    total_jobs: int
    fetched_count: int
    current_offset: int
    job_uuids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update(self, new_uuids: list[str], new_offset: int) -> None:
        """Update checkpoint with new progress."""
        self.job_uuids.extend(new_uuids)
        self.fetched_count = len(self.job_uuids)
        self.current_offset = new_offset
        self.updated_at = datetime.now()
