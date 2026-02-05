"""
Test data factories for generating realistic job data.

These factories create valid Pydantic models that mirror real MCF API responses,
making tests more reliable and easier to maintain.
"""

import random
import uuid
from datetime import date, timedelta
from typing import Optional

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


# =============================================================================
# Sample Data Pools
# =============================================================================

TITLES = [
    "Data Scientist",
    "Machine Learning Engineer",
    "Software Engineer",
    "Data Analyst",
    "Backend Developer",
    "DevOps Engineer",
    "Product Manager",
    "UX Designer",
    "QA Engineer",
    "Full Stack Developer",
    "Data Engineer",
    "AI Researcher",
    "Cloud Architect",
    "Security Engineer",
    "Mobile Developer",
]

COMPANIES = [
    ("Google Asia Pacific", "T12345678A"),
    ("Meta Platforms Singapore", "T23456789B"),
    ("Amazon Web Services", "T34567890C"),
    ("Microsoft Singapore", "T45678901D"),
    ("Grab Holdings", "T56789012E"),
    ("Shopee Singapore", "T67890123F"),
    ("Sea Limited", "T78901234G"),
    ("DBS Bank", "T89012345H"),
    ("OCBC Bank", "T90123456I"),
    ("Singtel", "T01234567J"),
]

SKILLS_POOL = [
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "SQL",
    "AWS",
    "Machine Learning",
    "TensorFlow",
    "PyTorch",
    "Docker",
    "Kubernetes",
    "React",
    "Node.js",
    "PostgreSQL",
    "MongoDB",
    "Redis",
    "Git",
    "CI/CD",
    "Agile",
    "REST API",
    "GraphQL",
    "Linux",
    "Azure",
    "GCP",
    "Spark",
]

CATEGORIES = [
    "Information Technology",
    "Engineering",
    "Data Science",
    "Software Development",
    "Cloud Computing",
    "Artificial Intelligence",
]

DISTRICTS = [
    "Downtown Core",
    "Orchard",
    "Marina Bay",
    "Bukit Merah",
    "Jurong East",
    "Tampines",
    "Woodlands",
    "Ang Mo Kio",
]

REGIONS = ["Central", "North", "South", "East", "West"]

POSITION_LEVELS = [
    "Junior",
    "Senior",
    "Lead",
    "Manager",
    "Director",
    "Executive",
]

EMPLOYMENT_TYPES = ["Full Time", "Part Time", "Contract", "Temporary", "Internship"]

DESCRIPTION_TEMPLATES = [
    "We are looking for a talented {title} to join our team at {company}. "
    "The ideal candidate will have experience with {skills}. "
    "You will be responsible for developing and maintaining our core systems.",
    "{company} is seeking a {title} to help build innovative solutions. "
    "Required skills include {skills}. "
    "This is an exciting opportunity to work on cutting-edge technology.",
    "Join {company} as a {title}! "
    "We need someone with strong {skills} skills. "
    "You'll collaborate with cross-functional teams to deliver high-quality products.",
]


# =============================================================================
# Factory Functions
# =============================================================================


def generate_job_uuid() -> str:
    """Generate a random UUID for a job."""
    return str(uuid.uuid4())


def generate_salary(
    min_salary: Optional[int] = None,
    max_salary: Optional[int] = None,
    salary_type: str = "Monthly",
) -> Salary:
    """
    Generate a salary object with realistic ranges.

    Args:
        min_salary: Override minimum salary
        max_salary: Override maximum salary
        salary_type: Salary type (Monthly, Yearly, Hourly)
    """
    if min_salary is None:
        min_salary = random.randint(4000, 15000)
    if max_salary is None:
        max_salary = min_salary + random.randint(1000, 5000)

    return Salary(
        minimum=min_salary,
        maximum=max_salary,
        type=SalaryType(salaryType=salary_type),
    )


def generate_company(
    name: Optional[str] = None,
    uen: Optional[str] = None,
) -> Company:
    """Generate a company object."""
    if name is None:
        company_tuple = random.choice(COMPANIES)
        name = company_tuple[0]
        uen = uen or company_tuple[1]

    return Company(
        name=name,
        uen=uen,
        description=f"A leading company in the technology industry.",
    )


def generate_skills(n: Optional[int] = None) -> list[Skill]:
    """Generate a list of skill objects."""
    if n is None:
        n = random.randint(3, 8)

    selected = random.sample(SKILLS_POOL, min(n, len(SKILLS_POOL)))
    return [
        Skill(skill=s, isKeySkill=(i < 3))  # First 3 are key skills
        for i, s in enumerate(selected)
    ]


def generate_categories(n: int = 2) -> list[Category]:
    """Generate a list of category objects."""
    selected = random.sample(CATEGORIES, min(n, len(CATEGORIES)))
    return [Category(category=c, id=i + 1) for i, c in enumerate(selected)]


def generate_address() -> Address:
    """Generate a location address."""
    return Address(
        block=str(random.randint(1, 100)),
        street=f"{random.choice(['Marina', 'Orchard', 'Shenton', 'Robinson'])} Road",
        floor=str(random.randint(1, 50)).zfill(2),
        unit=str(random.randint(1, 20)).zfill(2),
        postalCode=str(random.randint(100000, 999999)),
        district=random.choice(DISTRICTS),
        region=random.choice(REGIONS),
    )


def generate_metadata(
    posted_days_ago: Optional[int] = None,
    applications: Optional[int] = None,
) -> JobMetadata:
    """Generate job metadata with posting date and applications count."""
    if posted_days_ago is None:
        posted_days_ago = random.randint(0, 60)
    if applications is None:
        applications = random.randint(0, 150)

    posting_date = date.today() - timedelta(days=posted_days_ago)
    expiry_date = posting_date + timedelta(days=30)

    return JobMetadata(
        totalNumberJobApplication=applications,
        newPostingDate=posting_date.isoformat(),
        originalPostingDate=posting_date.isoformat(),
        expiryDate=expiry_date.isoformat(),
        isPostedOnBehalf=False,
    )


def generate_test_job(
    title: Optional[str] = None,
    company_name: Optional[str] = None,
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    skills: Optional[list[str]] = None,
    employment_type: str = "Full Time",
    position_level: str = "Senior",
    job_uuid: Optional[str] = None,
) -> Job:
    """
    Generate a single test job with optional overrides.

    This creates a complete Job object with all nested models populated,
    making it suitable for database insertion and API testing.

    Args:
        title: Override job title
        company_name: Override company name
        salary_min: Override minimum salary
        salary_max: Override maximum salary
        skills: Override skills list (as strings)
        employment_type: Employment type string
        position_level: Position level string
        job_uuid: Override UUID

    Returns:
        Fully populated Job object
    """
    if job_uuid is None:
        job_uuid = generate_job_uuid()

    if title is None:
        title = random.choice(TITLES)

    company = generate_company(name=company_name)
    salary = generate_salary(min_salary=salary_min, max_salary=salary_max)

    if skills is not None:
        skill_objects = [Skill(skill=s, isKeySkill=False) for s in skills]
    else:
        skill_objects = generate_skills()

    # Build description from template
    skill_names = ", ".join(s.skill for s in skill_objects[:3])
    description = random.choice(DESCRIPTION_TEMPLATES).format(
        title=title,
        company=company.name,
        skills=skill_names,
    )

    return Job(
        uuid=job_uuid,
        title=title,
        description=description,
        salary=salary,
        postedCompany=company,
        skills=skill_objects,
        categories=generate_categories(),
        address=generate_address(),
        employmentTypes=[EmploymentType(employmentType=employment_type)],
        positionLevels=[PositionLevel(position=position_level)],
        minimumYearsExperience=random.randint(0, 10),
        metadata=generate_metadata(),
        numberOfVacancies=random.randint(1, 5),
    )


def generate_test_jobs(n: int = 10) -> list[Job]:
    """
    Generate n test jobs with varied data.

    Args:
        n: Number of jobs to generate

    Returns:
        List of Job objects
    """
    return [generate_test_job() for _ in range(n)]


def generate_similar_jobs(base_title: str, n: int = 5) -> list[Job]:
    """
    Generate n jobs with similar titles for testing similarity search.

    Creates variations like "Senior {base_title}", "Junior {base_title}", etc.
    All jobs share similar skills to test semantic similarity.

    Args:
        base_title: Base job title to vary
        n: Number of similar jobs to generate

    Returns:
        List of Job objects with related titles
    """
    variations = [
        f"Senior {base_title}",
        f"Junior {base_title}",
        f"{base_title} Lead",
        f"{base_title} Manager",
        f"Principal {base_title}",
        f"Staff {base_title}",
        f"{base_title} II",
        f"{base_title} III",
    ]

    # Use consistent skills for similarity
    base_skills = random.sample(SKILLS_POOL, 5)

    jobs = []
    for i in range(n):
        title = variations[i % len(variations)]
        # Add some skill variation but keep core skills
        job_skills = base_skills[:3] + random.sample(SKILLS_POOL, 2)
        jobs.append(generate_test_job(title=title, skills=job_skills))

    return jobs


def generate_company_job_set(company_name: str, n: int = 10) -> list[Job]:
    """
    Generate n jobs all from the same company.

    Useful for testing company-based grouping and statistics.

    Args:
        company_name: Company name for all jobs
        n: Number of jobs to generate

    Returns:
        List of Job objects from the same company
    """
    return [generate_test_job(company_name=company_name) for _ in range(n)]


def generate_salary_range_jobs(
    min_salary: int,
    max_salary: int,
    n: int = 10,
) -> list[Job]:
    """
    Generate jobs within a specific salary range.

    Useful for testing salary filters.

    Args:
        min_salary: Minimum salary floor
        max_salary: Maximum salary ceiling
        n: Number of jobs

    Returns:
        List of Job objects with salaries in range
    """
    jobs = []
    for _ in range(n):
        sal_min = random.randint(min_salary, max_salary - 1000)
        sal_max = sal_min + random.randint(500, min(2000, max_salary - sal_min))
        jobs.append(generate_test_job(salary_min=sal_min, salary_max=sal_max))
    return jobs
