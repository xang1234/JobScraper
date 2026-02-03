"""
Storage layer for scraped job data.

Provides:
- CSV and JSON output formats
- Deduplication by job UUID
- Checkpoint save/restore for resumable scraping
- SQLite storage with history tracking
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import Job, Checkpoint
from .database import MCFDatabase

logger = logging.getLogger(__name__)


class JobStorage:
    """
    Manages storage of scraped job data with deduplication.

    Features:
    - Automatic deduplication by UUID
    - Multiple output formats (CSV, JSON)
    - Checkpoint support for resume functionality
    - Append mode for incremental updates

    Example:
        storage = JobStorage(output_dir="data")
        storage.add_jobs(jobs)
        storage.save_csv("data_scientist")
    """

    def __init__(
        self,
        output_dir: str = "data",
        checkpoint_dir: str = ".mcf_checkpoints",
    ):
        """
        Initialize job storage.

        Args:
            output_dir: Directory for output files (CSV, JSON)
            checkpoint_dir: Directory for checkpoint files
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self._jobs: dict[str, Job] = {}  # UUID -> Job mapping for dedup

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def job_count(self) -> int:
        """Number of unique jobs stored."""
        return len(self._jobs)

    @property
    def jobs(self) -> list[Job]:
        """Get all stored jobs as a list."""
        return list(self._jobs.values())

    def add_job(self, job: Job) -> bool:
        """
        Add a job to storage.

        Args:
            job: Job to add

        Returns:
            True if job was new, False if it was a duplicate
        """
        if job.uuid in self._jobs:
            logger.debug(f"Duplicate job skipped: {job.uuid}")
            return False

        self._jobs[job.uuid] = job
        return True

    def add_jobs(self, jobs: list[Job]) -> tuple[int, int]:
        """
        Add multiple jobs to storage.

        Args:
            jobs: List of jobs to add

        Returns:
            Tuple of (new_count, duplicate_count)
        """
        new_count = 0
        dup_count = 0

        for job in jobs:
            if self.add_job(job):
                new_count += 1
            else:
                dup_count += 1

        return new_count, dup_count

    def has_job(self, uuid: str) -> bool:
        """Check if a job with given UUID is already stored."""
        return uuid in self._jobs

    def get_uuids(self) -> set[str]:
        """Get set of all stored job UUIDs."""
        return set(self._jobs.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert stored jobs to a pandas DataFrame.

        Returns:
            DataFrame with flattened job data
        """
        if not self._jobs:
            return pd.DataFrame()

        records = [job.to_flat_dict() for job in self._jobs.values()]
        return pd.DataFrame(records)

    def save_csv(
        self,
        search_query: str,
        include_date: bool = True,
    ) -> Path:
        """
        Save jobs to CSV file.

        Args:
            search_query: Search term used (for filename)
            include_date: Whether to include date in filename

        Returns:
            Path to the saved CSV file
        """
        # Build filename
        safe_query = search_query.lower().replace(" ", "_")
        if include_date:
            filename = f"mcf_{safe_query}_{date.today().isoformat()}.csv"
        else:
            filename = f"mcf_{safe_query}.csv"

        filepath = self.output_dir / filename

        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

        logger.info(f"Saved {len(df)} jobs to {filepath}")
        return filepath

    def save_json(
        self,
        search_query: str,
        include_date: bool = True,
    ) -> Path:
        """
        Save jobs to JSON file.

        Args:
            search_query: Search term used (for filename)
            include_date: Whether to include date in filename

        Returns:
            Path to the saved JSON file
        """
        safe_query = search_query.lower().replace(" ", "_")
        if include_date:
            filename = f"mcf_{safe_query}_{date.today().isoformat()}.json"
        else:
            filename = f"mcf_{safe_query}.json"

        filepath = self.output_dir / filename

        # Convert jobs to JSON-serializable dicts
        data = {
            "query": search_query,
            "scraped_at": datetime.now().isoformat(),
            "total_jobs": len(self._jobs),
            "jobs": [job.to_flat_dict() for job in self._jobs.values()],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved {len(self._jobs)} jobs to {filepath}")
        return filepath

    def load_existing_csv(self, filepath: Path) -> int:
        """
        Load jobs from an existing CSV file for deduplication.

        Args:
            filepath: Path to existing CSV file

        Returns:
            Number of UUIDs loaded
        """
        if not filepath.exists():
            return 0

        try:
            df = pd.read_csv(filepath)
            if "uuid" in df.columns:
                for uuid in df["uuid"].dropna():
                    self._jobs[str(uuid)] = None  # Mark as seen
                logger.info(f"Loaded {len(df)} existing job UUIDs from {filepath}")
                return len(df)
        except Exception as e:
            logger.warning(f"Failed to load existing CSV: {e}")

        return 0

    # Checkpoint methods for resume functionality

    def _checkpoint_path(self, search_query: str) -> Path:
        """Get checkpoint file path for a search query."""
        safe_query = search_query.lower().replace(" ", "_")
        return self.checkpoint_dir / f"checkpoint_{safe_query}.json"

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        Save a checkpoint for resumable scraping.

        Args:
            checkpoint: Checkpoint data to save
        """
        filepath = self._checkpoint_path(checkpoint.search_query)

        with open(filepath, "w") as f:
            json.dump(checkpoint.model_dump(mode="json"), f, indent=2, default=str)

        logger.debug(f"Checkpoint saved: {checkpoint.fetched_count} jobs at offset {checkpoint.current_offset}")

    def load_checkpoint(self, search_query: str) -> Optional[Checkpoint]:
        """
        Load a checkpoint if one exists.

        Args:
            search_query: The search query to load checkpoint for

        Returns:
            Checkpoint if found, None otherwise
        """
        filepath = self._checkpoint_path(search_query)

        if not filepath.exists():
            return None

        try:
            with open(filepath) as f:
                data = json.load(f)
            checkpoint = Checkpoint.model_validate(data)
            logger.info(
                f"Resumed from checkpoint: {checkpoint.fetched_count} jobs, "
                f"offset {checkpoint.current_offset}"
            )
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self, search_query: str) -> None:
        """
        Remove checkpoint file after successful completion.

        Args:
            search_query: The search query whose checkpoint to remove
        """
        filepath = self._checkpoint_path(search_query)
        if filepath.exists():
            filepath.unlink()
            logger.debug(f"Checkpoint cleared for '{search_query}'")

    def clear_all(self) -> None:
        """Clear all stored jobs (for testing or fresh start)."""
        self._jobs.clear()


class SQLiteStorage:
    """
    SQLite-backed storage with history tracking.

    This class provides the same interface as JobStorage but persists
    data to SQLite with automatic deduplication and history tracking.

    Features:
    - Persistent storage across sessions
    - Automatic deduplication by UUID
    - History tracking when jobs are updated
    - Session-based progress tracking
    - Export to CSV/JSON

    Example:
        storage = SQLiteStorage()
        storage.add_job(job)  # Returns True if new
        storage.save_csv("data_scientist")
    """

    def __init__(
        self,
        db_path: str = "data/mcf_jobs.db",
        output_dir: str = "data",
    ):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
            output_dir: Directory for CSV/JSON exports
        """
        self.db = MCFDatabase(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for current session (fast duplicate checks)
        self._session_jobs: set[str] = set()
        self._session_id: Optional[int] = None
        self._current_query: Optional[str] = None

        # Track jobs added in this session for the jobs property
        self._jobs_this_session: dict[str, Job] = {}

    @property
    def job_count(self) -> int:
        """Number of unique jobs in database."""
        return self.db.count_jobs()

    @property
    def session_job_count(self) -> int:
        """Number of jobs added in current session."""
        return len(self._session_jobs)

    @property
    def jobs(self) -> list[Job]:
        """Get jobs added in current session."""
        return list(self._jobs_this_session.values())

    def add_job(self, job: Job) -> bool:
        """
        Add a job to storage.

        Args:
            job: Job to add

        Returns:
            True if job was new, False if it was a duplicate
        """
        # Fast check in session cache
        if job.uuid in self._session_jobs:
            logger.debug(f"Duplicate job in session: {job.uuid}")
            return False

        # Check/add to database
        is_new, was_updated = self.db.upsert_job(job)

        # Update session cache
        self._session_jobs.add(job.uuid)
        self._jobs_this_session[job.uuid] = job

        if is_new:
            logger.debug(f"New job added: {job.uuid}")
        elif was_updated:
            logger.debug(f"Job updated: {job.uuid}")

        return is_new

    def add_jobs(self, jobs: list[Job]) -> tuple[int, int]:
        """
        Add multiple jobs to storage.

        Args:
            jobs: List of jobs to add

        Returns:
            Tuple of (new_count, duplicate_count)
        """
        new_count = 0
        dup_count = 0

        for job in jobs:
            if self.add_job(job):
                new_count += 1
            else:
                dup_count += 1

        return new_count, dup_count

    def has_job(self, uuid: str) -> bool:
        """Check if a job with given UUID exists."""
        # Check session cache first (fast path)
        if uuid in self._session_jobs:
            return True
        # Fall back to database
        return self.db.has_job(uuid)

    def get_uuids(self) -> set[str]:
        """Get set of all job UUIDs in database."""
        return self.db.get_all_uuids()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert session jobs to a pandas DataFrame.

        Returns:
            DataFrame with flattened job data
        """
        if not self._jobs_this_session:
            return pd.DataFrame()

        records = [job.to_flat_dict() for job in self._jobs_this_session.values()]
        return pd.DataFrame(records)

    def save_csv(
        self,
        search_query: str,
        include_date: bool = True,
    ) -> Path:
        """
        Save session jobs to CSV file.

        Args:
            search_query: Search term used (for filename)
            include_date: Whether to include date in filename

        Returns:
            Path to the saved CSV file
        """
        safe_query = search_query.lower().replace(" ", "_")
        if include_date:
            filename = f"mcf_{safe_query}_{date.today().isoformat()}.csv"
        else:
            filename = f"mcf_{safe_query}.csv"

        filepath = self.output_dir / filename

        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

        logger.info(f"Saved {len(df)} jobs to {filepath}")
        return filepath

    def save_json(
        self,
        search_query: str,
        include_date: bool = True,
    ) -> Path:
        """
        Save session jobs to JSON file.

        Args:
            search_query: Search term used (for filename)
            include_date: Whether to include date in filename

        Returns:
            Path to the saved JSON file
        """
        safe_query = search_query.lower().replace(" ", "_")
        if include_date:
            filename = f"mcf_{safe_query}_{date.today().isoformat()}.json"
        else:
            filename = f"mcf_{safe_query}.json"

        filepath = self.output_dir / filename

        data = {
            "query": search_query,
            "scraped_at": datetime.now().isoformat(),
            "total_jobs": len(self._jobs_this_session),
            "jobs": [job.to_flat_dict() for job in self._jobs_this_session.values()],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved {len(self._jobs_this_session)} jobs to {filepath}")
        return filepath

    # Session/checkpoint methods

    def start_session(self, search_query: str, total_jobs: int) -> None:
        """
        Start a new scrape session.

        Args:
            search_query: The search query being scraped
            total_jobs: Total jobs available
        """
        self._current_query = search_query
        self._session_id = self.db.create_session(search_query, total_jobs)
        logger.debug(f"Started session {self._session_id} for '{search_query}'")

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        Save scrape progress (checkpoint).

        Args:
            checkpoint: Checkpoint data to save
        """
        if self._session_id:
            self.db.update_session(
                self._session_id,
                checkpoint.fetched_count,
                checkpoint.current_offset,
            )
            logger.debug(
                f"Session {self._session_id} updated: "
                f"{checkpoint.fetched_count} jobs at offset {checkpoint.current_offset}"
            )

    def load_checkpoint(self, search_query: str) -> Optional[Checkpoint]:
        """
        Load checkpoint if one exists for the query.

        Args:
            search_query: The search query to load checkpoint for

        Returns:
            Checkpoint if found, None otherwise
        """
        session = self.db.get_incomplete_session(search_query)
        if not session:
            return None

        # Resume from this session
        self._session_id = session["id"]
        self._current_query = search_query

        checkpoint = Checkpoint(
            search_query=search_query,
            total_jobs=session["total_jobs"],
            fetched_count=session["fetched_count"],
            current_offset=session["current_offset"],
            job_uuids=[],  # We don't store UUIDs in session, but DB has all jobs
        )

        logger.info(
            f"Resumed session {self._session_id}: "
            f"{checkpoint.fetched_count} jobs, offset {checkpoint.current_offset}"
        )
        return checkpoint

    def clear_checkpoint(self, search_query: str) -> None:
        """
        Mark session as completed.

        Args:
            search_query: The search query whose session to complete
        """
        if self._session_id:
            self.db.complete_session(self._session_id)
            logger.debug(f"Session {self._session_id} completed")
            self._session_id = None

    def clear_all(self) -> None:
        """Clear session cache (for testing or fresh session)."""
        self._session_jobs.clear()
        self._jobs_this_session.clear()
        self._session_id = None
        self._current_query = None

    # Database query methods

    def search(self, **filters) -> list[dict]:
        """
        Search jobs in database.

        Args:
            **filters: Search filters (keyword, company_name, salary_min, etc.)

        Returns:
            List of matching job dicts
        """
        return self.db.search_jobs(**filters)

    def get_stats(self) -> dict:
        """Get database statistics."""
        return self.db.get_stats()

    def get_job(self, uuid: str) -> Optional[dict]:
        """Get a job by UUID."""
        return self.db.get_job(uuid)

    def get_job_history(self, uuid: str) -> list[dict]:
        """Get history records for a job."""
        return self.db.get_job_history(uuid)

    def export_csv(self, output_path: Path, **filters) -> int:
        """
        Export jobs from database to CSV.

        Args:
            output_path: Path for output CSV
            **filters: Filters to apply

        Returns:
            Number of jobs exported
        """
        return self.db.export_to_csv(output_path, **filters)
