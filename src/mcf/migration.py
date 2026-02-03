"""
Legacy data migration for MCF job data.

Migrates job records from legacy formats (JSON files, CSVs) into the SQLite database.
Handles deduplication, field normalization, and tracks migration statistics.

Usage:
    from src.mcf.migration import MCFMigrator
    migrator = MCFMigrator("data/mcf_jobs.db")
    stats = migrator.migrate_all("data")
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .database import MCFDatabase
from .models import (
    Job,
    Salary,
    SalaryType,
    Company,
    Address,
    EmploymentType,
    PositionLevel,
    JobMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class MigrationError:
    """Record of a migration error."""
    source: str  # File path or identifier
    row: Optional[int]  # Row number if applicable
    error: str  # Error message
    data: Optional[dict] = None  # Raw data that caused the error


@dataclass
class MigrationStats:
    """Statistics from a migration run."""
    json_files_processed: int = 0
    csv_rows_processed: int = 0
    new_jobs: int = 0
    updated_jobs: int = 0
    skipped_duplicates: int = 0
    link_only_jobs: int = 0
    errors: list[MigrationError] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        """Total items processed."""
        return self.json_files_processed + self.csv_rows_processed

    @property
    def total_imported(self) -> int:
        """Total jobs imported (new + updated)."""
        return self.new_jobs + self.updated_jobs

    def __str__(self) -> str:
        return (
            f"Migration Stats:\n"
            f"  JSON files processed: {self.json_files_processed:,}\n"
            f"  CSV rows processed: {self.csv_rows_processed:,}\n"
            f"  New jobs: {self.new_jobs:,}\n"
            f"  Updated jobs: {self.updated_jobs:,}\n"
            f"  Skipped duplicates: {self.skipped_duplicates:,}\n"
            f"  Link-only jobs: {self.link_only_jobs:,}\n"
            f"  Errors: {len(self.errors):,}"
        )


class LegacyJobParser:
    """Parse and transform legacy job data formats into Job models."""

    # UUID pattern: 32 hex characters
    UUID_PATTERN = re.compile(r"([a-f0-9]{32})")

    # Salary pattern: extracts number from "$10,000" or "to$10,000"
    SALARY_PATTERN = re.compile(r"\$?([\d,]+)")

    # Experience pattern: extracts years from "3 years exp" or "3 year exp"
    EXPERIENCE_PATTERN = re.compile(r"(\d+)\s*years?\s*exp", re.IGNORECASE)

    # Applications pattern: extracts number from "22 applications"
    APPLICATIONS_PATTERN = re.compile(r"(\d+)\s*applications?", re.IGNORECASE)

    # Date pattern: "Posted 26 May 2023" or "Closing on 09 Nov 2023"
    DATE_PATTERN = re.compile(
        r"(?:Posted|Closing on)\s+(\d{1,2})\s+(\w+)\s+(\d{4})", re.IGNORECASE
    )

    MONTH_MAP = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    @classmethod
    def extract_uuid_from_url(cls, url: str) -> Optional[str]:
        """
        Extract 32-char hex UUID from MCF URL.

        Example URL:
            https://www.mycareersfuture.gov.sg/job/banking-finance/vice-president-dbs-bank-0011b2c1df8179f0f490d1c5316c7adc

        Returns the UUID or None if not found.
        """
        if not url:
            return None
        match = cls.UUID_PATTERN.search(url)
        return match.group(1) if match else None

    @classmethod
    def extract_uuid_from_filename(cls, filename: str) -> Optional[str]:
        """
        Extract UUID from JSON filename.

        Filename format: 0011b2c1df8179f0f490d1c5316c7adc.json
        """
        if not filename:
            return None
        name = Path(filename).stem
        if cls.UUID_PATTERN.fullmatch(name):
            return name
        return None

    @classmethod
    def parse_salary(cls, salary_str: str) -> Optional[int]:
        """
        Parse salary string to integer.

        Examples:
            "$10,000" -> 10000
            "to$18,700" -> 18700
            "$3,500" -> 3500
        """
        if not salary_str:
            return None
        match = cls.SALARY_PATTERN.search(str(salary_str))
        if match:
            return int(match.group(1).replace(",", ""))
        return None

    @classmethod
    def parse_experience(cls, exp_str: str) -> Optional[int]:
        """
        Parse experience string to years.

        Examples:
            "3 years exp" -> 3
            "15 years exp" -> 15
            "1 year exp" -> 1
        """
        if not exp_str:
            return None
        match = cls.EXPERIENCE_PATTERN.search(str(exp_str))
        return int(match.group(1)) if match else None

    @classmethod
    def parse_applications(cls, app_str: str) -> Optional[int]:
        """
        Parse applications count string.

        Examples:
            "22 applications" -> 22
            "7 applications" -> 7
        """
        if not app_str:
            return None
        match = cls.APPLICATIONS_PATTERN.search(str(app_str))
        return int(match.group(1)) if match else None

    @classmethod
    def parse_legacy_date(cls, date_str: str) -> Optional[date]:
        """
        Parse legacy date string.

        Examples:
            "Posted 26 May 2023" -> date(2023, 5, 26)
            "Closing on 09 Nov 2023" -> date(2023, 11, 9)
        """
        if not date_str:
            return None

        match = cls.DATE_PATTERN.search(str(date_str))
        if not match:
            return None

        day = int(match.group(1))
        month_str = match.group(2).lower()
        year = int(match.group(3))

        month = cls.MONTH_MAP.get(month_str)
        if not month:
            return None

        try:
            return date(year, month, day)
        except ValueError:
            return None

    @classmethod
    def build_job_from_json(cls, uuid: str, data: dict) -> Job:
        """
        Build a Job model from legacy JSON data.

        JSON fields:
            company, title, employment type, location, level, min experience,
            category, min salary, max salary, salary type, number of applications,
            last posted date, expiry date, job description, company info, link
        """
        # Parse salary
        salary = Salary(
            minimum=cls.parse_salary(data.get("min salary")),
            maximum=cls.parse_salary(data.get("max salary")),
            type=SalaryType(salaryType=data.get("salary type", "Monthly")),
        )

        # Parse company
        company = Company(
            name=data.get("company", ""),
            description=data.get("company info"),
        )

        # Parse address
        location = data.get("location", "")
        address = Address(street=location if location else None)

        # Parse employment type
        emp_type_str = data.get("employment type", "")
        # Format: "Permanent, Full Time" -> use first part for type
        emp_parts = [p.strip() for p in emp_type_str.split(",")]
        employment_types = [EmploymentType(employmentType=emp_parts[0])] if emp_parts else []

        # Parse position level
        level = data.get("level", "")
        position_levels = [PositionLevel(position=level)] if level else []

        # Parse metadata
        posted_date = cls.parse_legacy_date(data.get("last posted date"))
        expiry_date = cls.parse_legacy_date(data.get("expiry date"))
        applications = cls.parse_applications(data.get("number of applications"))

        metadata = JobMetadata(
            totalNumberJobApplication=applications or 0,
            newPostingDate=posted_date.isoformat() if posted_date else None,
            expiryDate=expiry_date.isoformat() if expiry_date else None,
        )

        return Job(
            uuid=uuid,
            title=data.get("title", ""),
            description=data.get("job description", ""),
            salary=salary,
            postedCompany=company,
            address=address,
            employmentTypes=employment_types,
            positionLevels=position_levels,
            minimumYearsExperience=cls.parse_experience(data.get("min experience")),
            metadata=metadata,
        )

    @classmethod
    def build_job_from_csv_row(cls, row: dict) -> Optional[Job]:
        """
        Build a Job model from a CSV row (full CSV format).

        CSV columns (case-insensitive):
            Company, Title, Location, Employment Type, Level, Minimum Experience,
            Category, Min Salary, Max Salary, Salary Type, Number of Applications,
            Last Posted, Closing Date, Job Description, Company Information, Link
        """
        # Normalize column names to lowercase
        row = {k.lower().strip(): v for k, v in row.items()}

        # Extract UUID from link
        link = row.get("link", "")
        uuid = cls.extract_uuid_from_url(link)
        if not uuid:
            return None

        title = row.get("title", "")
        if not title:
            return None

        # Parse salary
        salary = Salary(
            minimum=cls.parse_salary(row.get("min salary")),
            maximum=cls.parse_salary(row.get("max salary")),
            type=SalaryType(salaryType=row.get("salary type", "Monthly")),
        )

        # Parse company
        company = Company(
            name=row.get("company", ""),
            description=row.get("company information"),
        )

        # Parse address
        location = row.get("location", "")
        address = Address(street=location if location else None)

        # Parse employment type
        emp_type_str = row.get("employment type", "")
        emp_parts = [p.strip() for p in emp_type_str.split(",")] if emp_type_str else []
        employment_types = [EmploymentType(employmentType=emp_parts[0])] if emp_parts else []

        # Parse position level
        level = row.get("level", "")
        position_levels = [PositionLevel(position=level)] if level else []

        # Parse metadata
        posted_date = cls.parse_legacy_date(row.get("last posted"))
        expiry_date = cls.parse_legacy_date(row.get("closing date"))
        applications = cls.parse_applications(row.get("number of applications"))

        metadata = JobMetadata(
            totalNumberJobApplication=applications or 0,
            newPostingDate=posted_date.isoformat() if posted_date else None,
            expiryDate=expiry_date.isoformat() if expiry_date else None,
        )

        return Job(
            uuid=uuid,
            title=title,
            description=row.get("job description", ""),
            salary=salary,
            postedCompany=company,
            address=address,
            employmentTypes=employment_types,
            positionLevels=position_levels,
            minimumYearsExperience=cls.parse_experience(row.get("minimum experience")),
            metadata=metadata,
        )

    @classmethod
    def build_job_from_link_only(cls, row: dict) -> Optional[Job]:
        """
        Build a minimal Job model from link-only CSV.

        Link-only CSV columns:
            Company, Title, Location, Link
        """
        # Normalize column names to lowercase
        row = {k.lower().strip(): v for k, v in row.items()}

        link = row.get("link", "")
        uuid = cls.extract_uuid_from_url(link)
        if not uuid:
            return None

        title = row.get("title", "")
        if not title:
            return None

        company = Company(name=row.get("company", ""))
        location = row.get("location", "")
        address = Address(street=location if location else None)

        return Job(
            uuid=uuid,
            title=title,
            description="",
            postedCompany=company,
            address=address,
        )


class MCFMigrator:
    """
    Orchestrates migration of legacy MCF data into SQLite.

    Processes data sources in order for best data quality:
    1. JSON files first (richest data, filename IS the UUID)
    2. Full CSVs second (complete data, UUID from link)
    3. Link-only CSVs last (minimal data, only if UUID not seen)
    """

    def __init__(self, db_path: str = "data/mcf_jobs.db"):
        """
        Initialize migrator.

        Args:
            db_path: Path to SQLite database
        """
        self.db = MCFDatabase(db_path)
        self.parser = LegacyJobParser()
        self._seen_uuids: set[str] = set()

    def migrate_all(
        self,
        data_dir: str = "data",
        json_only: bool = False,
        csv_only: bool = False,
        skip_link_only: bool = False,
        dry_run: bool = False,
    ) -> MigrationStats:
        """
        Migrate all legacy data to SQLite.

        Args:
            data_dir: Base data directory
            json_only: Only process JSON files
            csv_only: Only process CSV files
            skip_link_only: Skip link-only CSVs (minimal data)
            dry_run: Preview without importing

        Returns:
            MigrationStats with counts and errors
        """
        stats = MigrationStats()
        data_path = Path(data_dir)

        # Load existing UUIDs to track what's already in DB
        if not dry_run:
            self._seen_uuids = self.db.get_all_uuids()
            logger.info(f"Database has {len(self._seen_uuids):,} existing jobs")

        # Process in order: JSON -> Full CSV -> Link-only CSV
        if not csv_only:
            self._migrate_json_files(data_path, stats, dry_run)

        if not json_only:
            self._migrate_full_csvs(data_path, stats, dry_run)

            if not skip_link_only:
                self._migrate_link_only_csvs(data_path, stats, dry_run)

        return stats

    def _migrate_json_files(
        self, data_path: Path, stats: MigrationStats, dry_run: bool
    ) -> None:
        """Process JSON files from scrape_jsons directory."""
        json_dir = data_path / "scrape_jsons"
        if not json_dir.exists():
            logger.warning(f"JSON directory not found: {json_dir}")
            return

        json_files = list(json_dir.glob("*.json"))
        logger.info(f"Found {len(json_files):,} JSON files to process")

        for json_file in json_files:
            try:
                uuid = self.parser.extract_uuid_from_filename(json_file.name)
                if not uuid:
                    stats.errors.append(MigrationError(
                        source=str(json_file),
                        row=None,
                        error="Could not extract UUID from filename",
                    ))
                    continue

                # Skip if already seen
                if uuid in self._seen_uuids:
                    stats.skipped_duplicates += 1
                    stats.json_files_processed += 1
                    continue

                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                job = self.parser.build_job_from_json(uuid, data)

                if not dry_run:
                    is_new, was_updated = self.db.upsert_job(job)
                    if is_new:
                        stats.new_jobs += 1
                    elif was_updated:
                        stats.updated_jobs += 1
                    else:
                        stats.skipped_duplicates += 1
                else:
                    stats.new_jobs += 1

                self._seen_uuids.add(uuid)
                stats.json_files_processed += 1

            except Exception as e:
                stats.errors.append(MigrationError(
                    source=str(json_file),
                    row=None,
                    error=str(e),
                ))
                logger.debug(f"Error processing {json_file}: {e}")

        logger.info(f"Processed {stats.json_files_processed:,} JSON files")

    def _migrate_full_csvs(
        self, data_path: Path, stats: MigrationStats, dry_run: bool
    ) -> None:
        """Process full MCF CSV files (16 columns)."""
        # Find MCF CSVs (not link-only, not LinkedIn)
        csv_files = [
            f for f in data_path.glob("mycareersfuture*.csv")
            if "link" not in f.name.lower()
        ]

        if not csv_files:
            logger.warning("No full MCF CSV files found")
            return

        logger.info(f"Found {len(csv_files):,} full CSV files to process")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding="utf-8")
                logger.debug(f"Processing {csv_file.name}: {len(df)} rows")

                for idx, row in df.iterrows():
                    try:
                        job = self.parser.build_job_from_csv_row(row.to_dict())
                        if not job:
                            stats.errors.append(MigrationError(
                                source=str(csv_file),
                                row=idx,
                                error="Could not parse row (missing UUID or title)",
                            ))
                            continue

                        # Skip if already seen
                        if job.uuid in self._seen_uuids:
                            stats.skipped_duplicates += 1
                            stats.csv_rows_processed += 1
                            continue

                        if not dry_run:
                            is_new, was_updated = self.db.upsert_job(job)
                            if is_new:
                                stats.new_jobs += 1
                            elif was_updated:
                                stats.updated_jobs += 1
                            else:
                                stats.skipped_duplicates += 1
                        else:
                            stats.new_jobs += 1

                        self._seen_uuids.add(job.uuid)
                        stats.csv_rows_processed += 1

                    except Exception as e:
                        stats.errors.append(MigrationError(
                            source=str(csv_file),
                            row=idx,
                            error=str(e),
                        ))

            except Exception as e:
                stats.errors.append(MigrationError(
                    source=str(csv_file),
                    row=None,
                    error=f"Could not read CSV: {e}",
                ))
                logger.warning(f"Error reading {csv_file}: {e}")

        logger.info(f"Processed {stats.csv_rows_processed:,} CSV rows")

    def _migrate_link_only_csvs(
        self, data_path: Path, stats: MigrationStats, dry_run: bool
    ) -> None:
        """Process link-only CSV files (4 columns: Company, Title, Location, Link)."""
        csv_files = list(data_path.glob("mycareersfuture*link*.csv"))

        if not csv_files:
            logger.info("No link-only CSV files found")
            return

        logger.info(f"Found {len(csv_files):,} link-only CSV files to process")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding="utf-8")
                logger.debug(f"Processing {csv_file.name}: {len(df)} rows")

                for idx, row in df.iterrows():
                    try:
                        job = self.parser.build_job_from_link_only(row.to_dict())
                        if not job:
                            stats.errors.append(MigrationError(
                                source=str(csv_file),
                                row=idx,
                                error="Could not parse row (missing UUID or title)",
                            ))
                            continue

                        # Only insert if UUID not seen (link-only has minimal data)
                        if job.uuid in self._seen_uuids:
                            stats.skipped_duplicates += 1
                            stats.csv_rows_processed += 1
                            continue

                        if not dry_run:
                            is_new, was_updated = self.db.upsert_job(job)
                            if is_new:
                                stats.new_jobs += 1
                                stats.link_only_jobs += 1
                            elif was_updated:
                                stats.updated_jobs += 1
                            else:
                                stats.skipped_duplicates += 1
                        else:
                            stats.new_jobs += 1
                            stats.link_only_jobs += 1

                        self._seen_uuids.add(job.uuid)
                        stats.csv_rows_processed += 1

                    except Exception as e:
                        stats.errors.append(MigrationError(
                            source=str(csv_file),
                            row=idx,
                            error=str(e),
                        ))

            except Exception as e:
                stats.errors.append(MigrationError(
                    source=str(csv_file),
                    row=None,
                    error=f"Could not read CSV: {e}",
                ))
                logger.warning(f"Error reading {csv_file}: {e}")

        logger.info(
            f"Processed link-only CSVs: {stats.link_only_jobs:,} new minimal records"
        )
