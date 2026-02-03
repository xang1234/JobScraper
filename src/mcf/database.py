"""
SQLite database manager for MCF job data.

Provides persistent storage with:
- Automatic deduplication by job UUID
- History tracking when jobs are updated
- Scrape session tracking (replaces JSON checkpoints)
- Query and export capabilities
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Any

from .models import Job, Checkpoint

logger = logging.getLogger(__name__)

# SQL schema definitions
SCHEMA_SQL = """
-- Main jobs table: current state of each job
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    company_name TEXT,
    company_uen TEXT,
    description TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_type TEXT,
    employment_type TEXT,
    seniority TEXT,
    min_experience_years INTEGER,
    skills TEXT,
    categories TEXT,
    location TEXT,
    district TEXT,
    region TEXT,
    posted_date DATE,
    expiry_date DATE,
    applications_count INTEGER,
    job_url TEXT,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- History table: previous versions of jobs (when updated)
CREATE TABLE IF NOT EXISTS job_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_uuid TEXT NOT NULL,
    title TEXT,
    company_name TEXT,
    salary_min INTEGER,
    salary_max INTEGER,
    applications_count INTEGER,
    description TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_uuid) REFERENCES jobs(uuid)
);

-- Scrape sessions table: replaces JSON checkpoints
CREATE TABLE IF NOT EXISTS scrape_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_query TEXT NOT NULL,
    total_jobs INTEGER,
    fetched_count INTEGER DEFAULT 0,
    current_offset INTEGER DEFAULT 0,
    status TEXT DEFAULT 'in_progress',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_jobs_uuid ON jobs(uuid);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company_name);
CREATE INDEX IF NOT EXISTS idx_jobs_salary ON jobs(salary_min, salary_max);
CREATE INDEX IF NOT EXISTS idx_jobs_posted ON jobs(posted_date);
CREATE INDEX IF NOT EXISTS idx_jobs_employment ON jobs(employment_type);
CREATE INDEX IF NOT EXISTS idx_history_uuid ON job_history(job_uuid);
CREATE INDEX IF NOT EXISTS idx_sessions_query ON scrape_sessions(search_query);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON scrape_sessions(status);

-- Historical scrape progress table: tracks enumeration of job IDs by year
CREATE TABLE IF NOT EXISTS historical_scrape_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    start_seq INTEGER NOT NULL,
    current_seq INTEGER NOT NULL,
    end_seq INTEGER,
    jobs_found INTEGER DEFAULT 0,
    jobs_not_found INTEGER DEFAULT 0,
    consecutive_not_found INTEGER DEFAULT 0,
    status TEXT DEFAULT 'in_progress',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_historical_year ON historical_scrape_progress(year);
CREATE INDEX IF NOT EXISTS idx_historical_status ON historical_scrape_progress(status);

-- Fetch attempts table: tracks every job ID fetch for completeness verification
CREATE TABLE IF NOT EXISTS fetch_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    result TEXT NOT NULL,
    error_message TEXT,
    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(year, sequence)
);

CREATE INDEX IF NOT EXISTS idx_fetch_year_seq ON fetch_attempts(year, sequence);
CREATE INDEX IF NOT EXISTS idx_fetch_result ON fetch_attempts(result);

-- Daemon state table: tracks background process for wake detection
CREATE TABLE IF NOT EXISTS daemon_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    pid INTEGER,
    status TEXT DEFAULT 'stopped',
    last_heartbeat TIMESTAMP,
    started_at TIMESTAMP,
    current_year INTEGER,
    current_seq INTEGER
);

-- Initialize daemon state row if not exists
INSERT OR IGNORE INTO daemon_state (id, status) VALUES (1, 'stopped');
"""


class MCFDatabase:
    """
    SQLite database manager for job data.

    Handles all database operations including schema creation,
    job upserts with history tracking, and query operations.

    Example:
        db = MCFDatabase("data/mcf_jobs.db")
        is_new, was_updated = db.upsert_job(job)
        jobs = db.search_jobs(company_name="Google")
    """

    def __init__(self, db_path: str = "data/mcf_jobs.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._connection() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.debug(f"Database schema ensured at {self.db_path}")

    def upsert_job(self, job: Job) -> tuple[bool, bool]:
        """
        Insert or update a job record.

        If the job exists and has changes, the old version is saved to history.

        Args:
            job: Job to insert or update

        Returns:
            Tuple of (is_new, was_updated)
            - is_new: True if this was a new job
            - was_updated: True if existing job was updated (changes detected)
        """
        job_data = job.to_flat_dict()
        now = datetime.now().isoformat()

        with self._connection() as conn:
            # Check if job exists
            existing = conn.execute(
                "SELECT * FROM jobs WHERE uuid = ?", (job.uuid,)
            ).fetchone()

            if existing is None:
                # New job - insert
                self._insert_job(conn, job_data, now)
                logger.debug(f"Inserted new job: {job.uuid}")
                return (True, False)

            # Existing job - check for changes
            changes = self._detect_changes(dict(existing), job_data)

            if changes:
                # Save current state to history
                self._save_to_history(conn, existing)

                # Update job
                self._update_job(conn, job_data, now)
                logger.debug(f"Updated job {job.uuid}: {', '.join(changes)}")
                return (False, True)

            # No changes
            return (False, False)

    def _insert_job(
        self, conn: sqlite3.Connection, job_data: dict, timestamp: str
    ) -> None:
        """Insert a new job record."""
        conn.execute(
            """
            INSERT INTO jobs (
                uuid, title, company_name, company_uen, description,
                salary_min, salary_max, salary_type, employment_type,
                seniority, min_experience_years, skills, categories,
                location, district, region, posted_date, expiry_date,
                applications_count, job_url, first_seen_at, last_updated_at
            ) VALUES (
                :uuid, :title, :company_name, :company_uen, :description,
                :salary_min, :salary_max, :salary_type, :employment_type,
                :seniority, :min_experience_years, :skills, :categories,
                :location, :district, :region, :posted_date, :expiry_date,
                :applications_count, :job_url, :first_seen_at, :last_updated_at
            )
            """,
            {**job_data, "first_seen_at": timestamp, "last_updated_at": timestamp},
        )

    def _update_job(
        self, conn: sqlite3.Connection, job_data: dict, timestamp: str
    ) -> None:
        """Update an existing job record."""
        conn.execute(
            """
            UPDATE jobs SET
                title = :title,
                company_name = :company_name,
                company_uen = :company_uen,
                description = :description,
                salary_min = :salary_min,
                salary_max = :salary_max,
                salary_type = :salary_type,
                employment_type = :employment_type,
                seniority = :seniority,
                min_experience_years = :min_experience_years,
                skills = :skills,
                categories = :categories,
                location = :location,
                district = :district,
                region = :region,
                posted_date = :posted_date,
                expiry_date = :expiry_date,
                applications_count = :applications_count,
                job_url = :job_url,
                last_updated_at = :last_updated_at
            WHERE uuid = :uuid
            """,
            {**job_data, "last_updated_at": timestamp},
        )

    def _save_to_history(self, conn: sqlite3.Connection, existing: sqlite3.Row) -> None:
        """Save current job state to history table."""
        conn.execute(
            """
            INSERT INTO job_history (
                job_uuid, title, company_name, salary_min, salary_max,
                applications_count, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                existing["uuid"],
                existing["title"],
                existing["company_name"],
                existing["salary_min"],
                existing["salary_max"],
                existing["applications_count"],
                existing["description"],
            ),
        )

    def _detect_changes(self, existing: dict, new_data: dict) -> list[str]:
        """
        Detect which fields have changed between existing and new data.

        Returns list of changed field names.
        """
        # Fields we track for changes
        tracked_fields = [
            "title",
            "company_name",
            "salary_min",
            "salary_max",
            "applications_count",
            "description",
            "employment_type",
            "seniority",
        ]

        changes = []
        for field in tracked_fields:
            old_val = existing.get(field)
            new_val = new_data.get(field)

            # Normalize None comparisons
            if old_val != new_val:
                changes.append(field)

        return changes

    def get_job(self, uuid: str) -> Optional[dict]:
        """
        Get a job by UUID.

        Args:
            uuid: Job UUID

        Returns:
            Job data as dict, or None if not found
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE uuid = ?", (uuid,)
            ).fetchone()
            return dict(row) if row else None

    def get_job_history(self, uuid: str) -> list[dict]:
        """
        Get history records for a job.

        Args:
            uuid: Job UUID

        Returns:
            List of historical records, newest first
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_history
                WHERE job_uuid = ?
                ORDER BY recorded_at DESC
                """,
                (uuid,),
            ).fetchall()
            return [dict(row) for row in rows]

    def search_jobs(
        self,
        keyword: Optional[str] = None,
        company_name: Optional[str] = None,
        salary_min: Optional[int] = None,
        salary_max: Optional[int] = None,
        employment_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Search jobs with filters.

        Args:
            keyword: Search in title, description, skills
            company_name: Filter by company (partial match)
            salary_min: Minimum salary filter
            salary_max: Maximum salary filter
            employment_type: Filter by employment type
            limit: Max results to return
            offset: Results offset for pagination

        Returns:
            List of matching jobs
        """
        conditions = []
        params: list[Any] = []

        if keyword:
            conditions.append(
                "(title LIKE ? OR description LIKE ? OR skills LIKE ?)"
            )
            like_pattern = f"%{keyword}%"
            params.extend([like_pattern, like_pattern, like_pattern])

        if company_name:
            conditions.append("company_name LIKE ?")
            params.append(f"%{company_name}%")

        if salary_min is not None:
            conditions.append("salary_min >= ?")
            params.append(salary_min)

        if salary_max is not None:
            conditions.append("salary_max <= ?")
            params.append(salary_max)

        if employment_type:
            conditions.append("employment_type = ?")
            params.append(employment_type)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM jobs
                WHERE {where_clause}
                ORDER BY posted_date DESC, last_updated_at DESC
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            ).fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dict with various statistics
        """
        with self._connection() as conn:
            stats = {}

            # Total jobs
            stats["total_jobs"] = conn.execute(
                "SELECT COUNT(*) FROM jobs"
            ).fetchone()[0]

            # Jobs by employment type
            rows = conn.execute(
                """
                SELECT employment_type, COUNT(*) as count
                FROM jobs
                GROUP BY employment_type
                ORDER BY count DESC
                """
            ).fetchall()
            stats["by_employment_type"] = {row[0]: row[1] for row in rows}

            # Top companies
            rows = conn.execute(
                """
                SELECT company_name, COUNT(*) as count
                FROM jobs
                WHERE company_name IS NOT NULL AND company_name != ''
                GROUP BY company_name
                ORDER BY count DESC
                LIMIT 10
                """
            ).fetchall()
            stats["top_companies"] = {row[0]: row[1] for row in rows}

            # Salary ranges
            row = conn.execute(
                """
                SELECT
                    MIN(salary_min) as min_salary,
                    MAX(salary_max) as max_salary,
                    AVG(salary_min) as avg_min,
                    AVG(salary_max) as avg_max
                FROM jobs
                WHERE salary_min IS NOT NULL AND salary_max IS NOT NULL
                """
            ).fetchone()
            stats["salary_stats"] = {
                "min": row[0],
                "max": row[1],
                "avg_min": int(row[2]) if row[2] else None,
                "avg_max": int(row[3]) if row[3] else None,
            }

            # History count
            stats["history_records"] = conn.execute(
                "SELECT COUNT(*) FROM job_history"
            ).fetchone()[0]

            # Jobs with history
            stats["jobs_with_history"] = conn.execute(
                "SELECT COUNT(DISTINCT job_uuid) FROM job_history"
            ).fetchone()[0]

            # Recent activity
            stats["jobs_added_today"] = conn.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE DATE(first_seen_at) = DATE('now')
                """
            ).fetchone()[0]

            stats["jobs_updated_today"] = conn.execute(
                """
                SELECT COUNT(*) FROM jobs
                WHERE DATE(last_updated_at) = DATE('now')
                AND first_seen_at != last_updated_at
                """
            ).fetchone()[0]

            return stats

    def has_job(self, uuid: str) -> bool:
        """Check if a job with the given UUID exists."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE uuid = ? LIMIT 1", (uuid,)
            ).fetchone()
            return row is not None

    def get_all_uuids(self) -> set[str]:
        """Get set of all job UUIDs in database."""
        with self._connection() as conn:
            rows = conn.execute("SELECT uuid FROM jobs").fetchall()
            return {row[0] for row in rows}

    def count_jobs(self) -> int:
        """Get total number of jobs in database."""
        with self._connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    # Scrape session methods (replaces checkpoint functionality)

    def create_session(self, search_query: str, total_jobs: int) -> int:
        """
        Create a new scrape session.

        Args:
            search_query: The search query being scraped
            total_jobs: Total jobs available for this query

        Returns:
            Session ID
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scrape_sessions (search_query, total_jobs, status)
                VALUES (?, ?, 'in_progress')
                """,
                (search_query, total_jobs),
            )
            return cursor.lastrowid

    def update_session(
        self,
        session_id: int,
        fetched_count: int,
        current_offset: int,
    ) -> None:
        """Update session progress."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE scrape_sessions
                SET fetched_count = ?, current_offset = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (fetched_count, current_offset, session_id),
            )

    def complete_session(self, session_id: int) -> None:
        """Mark session as completed."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE scrape_sessions
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (session_id,),
            )

    def get_incomplete_session(self, search_query: str) -> Optional[dict]:
        """
        Get the most recent incomplete session for a query.

        Args:
            search_query: Search query to look up

        Returns:
            Session data if found, None otherwise
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM scrape_sessions
                WHERE search_query = ? AND status = 'in_progress'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (search_query,),
            ).fetchone()
            return dict(row) if row else None

    def get_all_sessions(self, status: Optional[str] = None) -> list[dict]:
        """
        Get all scrape sessions.

        Args:
            status: Filter by status (in_progress, completed, interrupted)

        Returns:
            List of session records
        """
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM scrape_sessions
                    WHERE status = ?
                    ORDER BY started_at DESC
                    """,
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM scrape_sessions
                    ORDER BY started_at DESC
                    """
                ).fetchall()
            return [dict(row) for row in rows]

    def clear_incomplete_sessions(self) -> int:
        """
        Mark all incomplete sessions as interrupted.

        Returns:
            Number of sessions updated
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE scrape_sessions
                SET status = 'interrupted'
                WHERE status = 'in_progress'
                """
            )
            return cursor.rowcount

    def export_to_csv(self, output_path: Path, **filters) -> int:
        """
        Export jobs to CSV file.

        Args:
            output_path: Path for output CSV
            **filters: Filters to apply (same as search_jobs)

        Returns:
            Number of jobs exported
        """
        import pandas as pd

        # Use search with high limit to get all matching jobs
        jobs = self.search_jobs(**filters, limit=1000000)

        if not jobs:
            return 0

        df = pd.DataFrame(jobs)

        # Remove internal columns
        columns_to_drop = ["id", "first_seen_at", "last_updated_at"]
        df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} jobs to {output_path}")
        return len(df)

    # Historical scrape progress methods

    def create_historical_session(
        self, year: int, start_seq: int, end_seq: Optional[int] = None
    ) -> int:
        """
        Create a new historical scrape session.

        Args:
            year: Year being scraped (e.g., 2023)
            start_seq: Starting sequence number
            end_seq: Ending sequence number (None if unknown)

        Returns:
            Session ID
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO historical_scrape_progress
                    (year, start_seq, current_seq, end_seq, status)
                VALUES (?, ?, ?, ?, 'in_progress')
                """,
                (year, start_seq, start_seq, end_seq),
            )
            return cursor.lastrowid

    def update_historical_progress(
        self,
        session_id: int,
        current_seq: int,
        jobs_found: int,
        jobs_not_found: int,
        consecutive_not_found: int = 0,
    ) -> None:
        """Update historical scrape progress."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE historical_scrape_progress
                SET current_seq = ?,
                    jobs_found = ?,
                    jobs_not_found = ?,
                    consecutive_not_found = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (current_seq, jobs_found, jobs_not_found, consecutive_not_found, session_id),
            )

    def complete_historical_session(self, session_id: int) -> None:
        """Mark historical session as completed."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE historical_scrape_progress
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (session_id,),
            )

    def get_incomplete_historical_session(self, year: int) -> Optional[dict]:
        """
        Get the most recent incomplete historical session for a year.

        Args:
            year: Year to look up

        Returns:
            Session data if found, None otherwise
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM historical_scrape_progress
                WHERE year = ? AND status = 'in_progress'
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (year,),
            ).fetchone()
            return dict(row) if row else None

    def get_all_historical_sessions(self, status: Optional[str] = None) -> list[dict]:
        """
        Get all historical scrape sessions.

        Args:
            status: Filter by status (in_progress, completed, interrupted)

        Returns:
            List of session records
        """
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM historical_scrape_progress
                    WHERE status = ?
                    ORDER BY year DESC, started_at DESC
                    """,
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM historical_scrape_progress
                    ORDER BY year DESC, started_at DESC
                    """
                ).fetchall()
            return [dict(row) for row in rows]

    def clear_incomplete_historical_sessions(self) -> int:
        """
        Mark all incomplete historical sessions as interrupted.

        Returns:
            Number of sessions updated
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE historical_scrape_progress
                SET status = 'interrupted'
                WHERE status = 'in_progress'
                """
            )
            return cursor.rowcount

    def get_historical_stats(self) -> dict:
        """
        Get statistics about historical scraping.

        Returns:
            Dict with statistics by year
        """
        with self._connection() as conn:
            # Jobs by year (based on posted_date)
            rows = conn.execute(
                """
                SELECT strftime('%Y', posted_date) as year, COUNT(*) as count
                FROM jobs
                WHERE posted_date IS NOT NULL
                GROUP BY year
                ORDER BY year DESC
                """
            ).fetchall()
            jobs_by_year = {row[0]: row[1] for row in rows if row[0]}

            # Session progress by year
            rows = conn.execute(
                """
                SELECT year,
                       SUM(jobs_found) as total_found,
                       SUM(jobs_not_found) as total_not_found,
                       MAX(current_seq) as max_seq_reached
                FROM historical_scrape_progress
                GROUP BY year
                ORDER BY year DESC
                """
            ).fetchall()
            scrape_progress = {
                row[0]: {
                    "jobs_found": row[1],
                    "jobs_not_found": row[2],
                    "max_seq_reached": row[3],
                }
                for row in rows
            }

            return {
                "jobs_by_year": jobs_by_year,
                "scrape_progress": scrape_progress,
            }

    # Fetch attempt logging methods

    def batch_insert_attempts(self, attempts: list[dict]) -> int:
        """
        Insert or replace batch of fetch attempts.

        Args:
            attempts: List of attempt dicts with keys:
                - year: int
                - sequence: int
                - result: str ('found', 'not_found', 'error', 'skipped')
                - error_message: str or None

        Returns:
            Number of attempts inserted
        """
        if not attempts:
            return 0

        with self._connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO fetch_attempts
                    (year, sequence, result, error_message, attempted_at)
                VALUES (:year, :sequence, :result, :error_message, CURRENT_TIMESTAMP)
                """,
                attempts,
            )
            return len(attempts)

    def get_missing_sequences(self, year: int) -> list[tuple[int, int]]:
        """
        Find gaps in attempted sequences for a year.

        Uses window functions to detect ranges of missing sequence numbers
        between the first and last attempted sequence.

        Args:
            year: Year to check for gaps

        Returns:
            List of (start_seq, end_seq) tuples representing gaps
        """
        with self._connection() as conn:
            # Get the bounds
            bounds = conn.execute(
                """
                SELECT MIN(sequence) as min_seq, MAX(sequence) as max_seq
                FROM fetch_attempts
                WHERE year = ?
                """,
                (year,),
            ).fetchone()

            if not bounds or bounds["min_seq"] is None:
                return []

            min_seq, max_seq = bounds["min_seq"], bounds["max_seq"]

            # Get all attempted sequences as a set
            rows = conn.execute(
                """
                SELECT sequence FROM fetch_attempts
                WHERE year = ? AND sequence BETWEEN ? AND ?
                ORDER BY sequence
                """,
                (year, min_seq, max_seq),
            ).fetchall()

            attempted = {row["sequence"] for row in rows}

            # Find gaps
            gaps = []
            gap_start = None

            for seq in range(min_seq, max_seq + 1):
                if seq not in attempted:
                    if gap_start is None:
                        gap_start = seq
                else:
                    if gap_start is not None:
                        gaps.append((gap_start, seq - 1))
                        gap_start = None

            # Handle trailing gap
            if gap_start is not None:
                gaps.append((gap_start, max_seq))

            return gaps

    def get_failed_attempts(self, year: int, limit: int = 10000) -> list[dict]:
        """
        Get all attempts with result='error' for retry.

        Args:
            year: Year to query
            limit: Maximum number of results

        Returns:
            List of attempt dicts with error details
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT year, sequence, result, error_message, attempted_at
                FROM fetch_attempts
                WHERE year = ? AND result = 'error'
                ORDER BY sequence
                LIMIT ?
                """,
                (year, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_attempt_stats(self, year: int) -> dict:
        """
        Get counts by result type for a year.

        Args:
            year: Year to get statistics for

        Returns:
            Dict with counts: found, not_found, error, skipped, total
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT result, COUNT(*) as count
                FROM fetch_attempts
                WHERE year = ?
                GROUP BY result
                """,
                (year,),
            ).fetchall()

            stats = {row["result"]: row["count"] for row in rows}
            stats["total"] = sum(stats.values())

            # Get sequence bounds
            bounds = conn.execute(
                """
                SELECT MIN(sequence) as min_seq, MAX(sequence) as max_seq
                FROM fetch_attempts
                WHERE year = ?
                """,
                (year,),
            ).fetchone()

            if bounds and bounds["min_seq"] is not None:
                stats["min_sequence"] = bounds["min_seq"]
                stats["max_sequence"] = bounds["max_seq"]
                stats["sequence_range"] = bounds["max_seq"] - bounds["min_seq"] + 1

            return stats

    def get_all_attempt_stats(self) -> dict[int, dict]:
        """
        Get attempt statistics for all years.

        Returns:
            Dict mapping year to stats dict
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT year, result, COUNT(*) as count
                FROM fetch_attempts
                GROUP BY year, result
                ORDER BY year
                """
            ).fetchall()

            stats: dict[int, dict] = {}
            for row in rows:
                year = row["year"]
                if year not in stats:
                    stats[year] = {}
                stats[year][row["result"]] = row["count"]

            # Add totals
            for year in stats:
                stats[year]["total"] = sum(stats[year].values())

            return stats

    # Daemon state methods

    def update_daemon_state(
        self,
        pid: int,
        status: str,
        current_year: int | None = None,
        current_seq: int | None = None,
    ) -> None:
        """
        Update daemon state in database.

        Args:
            pid: Process ID
            status: 'running', 'stopped', or 'sleeping'
            current_year: Current year being scraped
            current_seq: Current sequence being scraped
        """
        with self._connection() as conn:
            if status == 'running':
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET pid = ?, status = ?, started_at = CURRENT_TIMESTAMP,
                        last_heartbeat = CURRENT_TIMESTAMP,
                        current_year = ?, current_seq = ?
                    WHERE id = 1
                    """,
                    (pid, status, current_year, current_seq),
                )
            else:
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET pid = ?, status = ?, current_year = ?, current_seq = ?
                    WHERE id = 1
                    """,
                    (pid, status, current_year, current_seq),
                )

    def update_daemon_heartbeat(
        self,
        current_year: int | None = None,
        current_seq: int | None = None,
    ) -> None:
        """
        Update daemon heartbeat timestamp.

        Args:
            current_year: Current year being scraped
            current_seq: Current sequence being scraped
        """
        with self._connection() as conn:
            if current_year is not None:
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET last_heartbeat = CURRENT_TIMESTAMP,
                        current_year = ?, current_seq = ?
                    WHERE id = 1
                    """,
                    (current_year, current_seq),
                )
            else:
                conn.execute(
                    """
                    UPDATE daemon_state
                    SET last_heartbeat = CURRENT_TIMESTAMP
                    WHERE id = 1
                    """
                )

    def get_daemon_state(self) -> dict:
        """
        Get current daemon state.

        Returns:
            Dict with daemon state including:
            - pid, status, last_heartbeat, started_at
            - current_year, current_seq
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM daemon_state WHERE id = 1"
            ).fetchone()

            if row:
                return dict(row)
            return {
                "pid": None,
                "status": "stopped",
                "last_heartbeat": None,
                "started_at": None,
                "current_year": None,
                "current_seq": None,
            }

    def clear_daemon_state(self) -> None:
        """Reset daemon state to stopped."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE daemon_state
                SET pid = NULL, status = 'stopped',
                    last_heartbeat = NULL, started_at = NULL,
                    current_year = NULL, current_seq = NULL
                WHERE id = 1
                """
            )
