"""
SQLite database manager for MCF job data.

Provides persistent storage with:
- Automatic deduplication by job UUID
- History tracking when jobs are updated
- Scrape session tracking (replaces JSON checkpoints)
- Query and export capabilities
- Embedding storage for semantic search
- FTS5 full-text search indexing
- Search analytics tracking
"""

import json
import logging
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Any

import numpy as np

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

# Schema for embeddings storage (semantic search)
EMBEDDINGS_SCHEMA = """
-- Embeddings table: stores vector embeddings for jobs, skills, companies
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL,           -- UUID for jobs, name for skills/companies
    entity_type TEXT NOT NULL,         -- 'job', 'skill', 'company'
    embedding_blob BLOB NOT NULL,      -- Serialized numpy array (384 × 4 = 1536 bytes)
    model_version TEXT DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(entity_id, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(entity_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_version);
"""

# Schema for FTS5 full-text search (external content table)
FTS5_SCHEMA = """
-- FTS5 virtual table for full-text search on jobs
-- Uses external content from jobs table (no data duplication)
CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
    uuid,
    title,
    description,
    skills,
    company_name,
    content='jobs',
    content_rowid='id'
);

-- Triggers to keep FTS in sync with jobs table
CREATE TRIGGER IF NOT EXISTS jobs_ai AFTER INSERT ON jobs BEGIN
    INSERT INTO jobs_fts(rowid, uuid, title, description, skills, company_name)
    VALUES (new.id, new.uuid, new.title, new.description, new.skills, new.company_name);
END;

CREATE TRIGGER IF NOT EXISTS jobs_ad AFTER DELETE ON jobs BEGIN
    INSERT INTO jobs_fts(jobs_fts, rowid, uuid, title, description, skills, company_name)
    VALUES ('delete', old.id, old.uuid, old.title, old.description, old.skills, old.company_name);
END;

CREATE TRIGGER IF NOT EXISTS jobs_au AFTER UPDATE ON jobs BEGIN
    INSERT INTO jobs_fts(jobs_fts, rowid, uuid, title, description, skills, company_name)
    VALUES ('delete', old.id, old.uuid, old.title, old.description, old.skills, old.company_name);
    INSERT INTO jobs_fts(rowid, uuid, title, description, skills, company_name)
    VALUES (new.id, new.uuid, new.title, new.description, new.skills, new.company_name);
END;
"""

# Schema for search analytics tracking
ANALYTICS_SCHEMA = """
-- Search analytics: tracks queries for monitoring and optimization
CREATE TABLE IF NOT EXISTS search_analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    query_type TEXT DEFAULT 'semantic',  -- 'semantic', 'keyword', 'hybrid'
    result_count INTEGER,
    latency_ms REAL,
    cache_hit BOOLEAN DEFAULT FALSE,
    degraded BOOLEAN DEFAULT FALSE,      -- True if fallback was used
    filters_used TEXT,                   -- JSON of applied filters
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analytics_time ON search_analytics(searched_at);
CREATE INDEX IF NOT EXISTS idx_analytics_query ON search_analytics(query);
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
        """Create tables and run migrations."""
        with self._connection() as conn:
            # Core schema (jobs, history, sessions, etc.)
            conn.executescript(SCHEMA_SQL)

            # Embeddings table
            conn.executescript(EMBEDDINGS_SCHEMA)

            # Search analytics table
            conn.executescript(ANALYTICS_SCHEMA)

        # Run migrations for schema changes to existing tables
        self._migrate_salary_annual()

        # FTS5 requires special handling (check if table exists first)
        self._ensure_fts5()

        logger.debug(f"Database schema ensured at {self.db_path}")

    def _migrate_salary_annual(self) -> None:
        """
        Add salary_annual columns if they don't exist.

        This handles the ALTER TABLE gracefully for existing databases.
        Normalizes all salaries to annual for consistent comparisons.
        """
        with self._connection() as conn:
            # Check if columns exist
            cursor = conn.execute("PRAGMA table_info(jobs)")
            columns = {row[1] for row in cursor.fetchall()}

            if "salary_annual_min" not in columns:
                logger.info("Adding salary_annual columns to jobs table...")
                conn.execute("ALTER TABLE jobs ADD COLUMN salary_annual_min INTEGER")
                conn.execute("ALTER TABLE jobs ADD COLUMN salary_annual_max INTEGER")

                # Populate for existing rows based on salary_type
                conn.execute("""
                    UPDATE jobs SET
                        salary_annual_min = CASE salary_type
                            WHEN 'Monthly' THEN salary_min * 12
                            WHEN 'Yearly' THEN salary_min
                            WHEN 'Hourly' THEN salary_min * 2080
                            WHEN 'Daily' THEN salary_min * 260
                            ELSE salary_min * 12  -- Assume monthly as default
                        END,
                        salary_annual_max = CASE salary_type
                            WHEN 'Monthly' THEN salary_max * 12
                            WHEN 'Yearly' THEN salary_max
                            WHEN 'Hourly' THEN salary_max * 2080
                            WHEN 'Daily' THEN salary_max * 260
                            ELSE salary_max * 12
                        END
                    WHERE salary_min IS NOT NULL OR salary_max IS NOT NULL
                """)
                conn.commit()
                logger.info("Salary annual migration complete")

    def _ensure_fts5(self) -> None:
        """
        Set up FTS5 virtual table and triggers.

        FTS5 tables need special handling because:
        1. They can't be created with IF NOT EXISTS in all cases
        2. Triggers need to be created separately
        3. Existing data needs to be indexed on first setup
        """
        with self._connection() as conn:
            # Check if FTS table exists
            fts_exists = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='jobs_fts'
            """).fetchone()

            if not fts_exists:
                logger.info("Creating FTS5 index for jobs...")
                conn.executescript(FTS5_SCHEMA)

                # Populate FTS5 with existing jobs data
                jobs_count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                if jobs_count > 0:
                    logger.info(f"Rebuilding FTS5 index for {jobs_count} jobs...")
                    conn.execute("INSERT INTO jobs_fts(jobs_fts) VALUES('rebuild')")

                conn.commit()
                logger.info("FTS5 index created")

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
        # Calculate annual salary for consistent comparisons
        data = {**job_data, "first_seen_at": timestamp, "last_updated_at": timestamp}
        data["salary_annual_min"], data["salary_annual_max"] = self._calculate_annual_salary(
            job_data.get("salary_min"),
            job_data.get("salary_max"),
            job_data.get("salary_type"),
        )

        conn.execute(
            """
            INSERT INTO jobs (
                uuid, title, company_name, company_uen, description,
                salary_min, salary_max, salary_type, employment_type,
                seniority, min_experience_years, skills, categories,
                location, district, region, posted_date, expiry_date,
                applications_count, job_url, first_seen_at, last_updated_at,
                salary_annual_min, salary_annual_max
            ) VALUES (
                :uuid, :title, :company_name, :company_uen, :description,
                :salary_min, :salary_max, :salary_type, :employment_type,
                :seniority, :min_experience_years, :skills, :categories,
                :location, :district, :region, :posted_date, :expiry_date,
                :applications_count, :job_url, :first_seen_at, :last_updated_at,
                :salary_annual_min, :salary_annual_max
            )
            """,
            data,
        )

    def _update_job(
        self, conn: sqlite3.Connection, job_data: dict, timestamp: str
    ) -> None:
        """Update an existing job record."""
        # Calculate annual salary for consistent comparisons
        data = {**job_data, "last_updated_at": timestamp}
        data["salary_annual_min"], data["salary_annual_max"] = self._calculate_annual_salary(
            job_data.get("salary_min"),
            job_data.get("salary_max"),
            job_data.get("salary_type"),
        )

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
                last_updated_at = :last_updated_at,
                salary_annual_min = :salary_annual_min,
                salary_annual_max = :salary_annual_max
            WHERE uuid = :uuid
            """,
            data,
        )

    @staticmethod
    def _calculate_annual_salary(
        salary_min: int | None,
        salary_max: int | None,
        salary_type: str | None,
    ) -> tuple[int | None, int | None]:
        """
        Convert salary to annual equivalent.

        Args:
            salary_min: Minimum salary in original units
            salary_max: Maximum salary in original units
            salary_type: 'Monthly', 'Yearly', 'Hourly', or 'Daily'

        Returns:
            Tuple of (annual_min, annual_max)
        """
        if salary_min is None and salary_max is None:
            return None, None

        # Conversion factors to annual
        multipliers = {
            "Monthly": 12,
            "Yearly": 1,
            "Hourly": 2080,  # 40 hours × 52 weeks
            "Daily": 260,   # 5 days × 52 weeks
        }
        multiplier = multipliers.get(salary_type, 12)  # Default to monthly

        annual_min = int(salary_min * multiplier) if salary_min else None
        annual_max = int(salary_max * multiplier) if salary_max else None

        return annual_min, annual_max

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

    # =========================================================================
    # Embedding Methods (Semantic Search)
    # =========================================================================

    def upsert_embedding(
        self,
        entity_id: str,
        entity_type: str,
        embedding: np.ndarray,
        model_version: str | None = None,
    ) -> None:
        """
        Insert or update an embedding.

        Args:
            entity_id: UUID for jobs, name for skills/companies
            entity_type: 'job', 'skill', or 'company'
            embedding: numpy array of shape (384,) for MiniLM
            model_version: Model used to generate embedding
        """
        blob = embedding.astype(np.float32).tobytes()
        model = model_version or "all-MiniLM-L6-v2"

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO embeddings (entity_id, entity_type, embedding_blob, model_version, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(entity_id, entity_type) DO UPDATE SET
                    embedding_blob = excluded.embedding_blob,
                    model_version = excluded.model_version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (entity_id, entity_type, blob, model),
            )

    def get_embedding(self, entity_id: str, entity_type: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding as numpy array.

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity ('job', 'skill', 'company')

        Returns:
            Embedding array or None if not found
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT embedding_blob FROM embeddings WHERE entity_id = ? AND entity_type = ?",
                (entity_id, entity_type),
            ).fetchone()

        if row:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

    def get_all_embeddings(self, entity_type: str) -> tuple[list[str], np.ndarray]:
        """
        Get all embeddings of a type as (IDs, stacked array).

        Useful for batch similarity calculations.

        Args:
            entity_type: 'job', 'skill', or 'company'

        Returns:
            Tuple of (entity_ids, embeddings_matrix)
            embeddings_matrix has shape (n_entities, embedding_dim)
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT entity_id, embedding_blob FROM embeddings WHERE entity_type = ? ORDER BY id",
                (entity_type,),
            ).fetchall()

        if not rows:
            return [], np.array([])

        ids = [row[0] for row in rows]
        embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        return ids, embeddings

    def get_embeddings_for_uuids(self, uuids: list[str]) -> dict[str, np.ndarray]:
        """
        Get embeddings for specific job UUIDs.

        Args:
            uuids: List of job UUIDs

        Returns:
            Dict mapping uuid -> embedding array
        """
        if not uuids:
            return {}

        placeholders = ",".join("?" * len(uuids))
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT entity_id, embedding_blob FROM embeddings "
                f"WHERE entity_type = 'job' AND entity_id IN ({placeholders})",
                uuids,
            ).fetchall()

        return {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in rows}

    def get_embedding_stats(self) -> dict:
        """
        Get embedding statistics including coverage.

        Returns:
            Dict with counts by type, coverage percentage, model version
        """
        with self._connection() as conn:
            # Count by type
            type_counts = {}
            for row in conn.execute(
                "SELECT entity_type, COUNT(*) FROM embeddings GROUP BY entity_type"
            ):
                type_counts[row[0]] = row[1]

            # Job coverage
            total_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            jobs_with_embeddings = type_counts.get("job", 0)

            # Model version
            model = conn.execute(
                "SELECT model_version FROM embeddings LIMIT 1"
            ).fetchone()

        return {
            "job_embeddings": type_counts.get("job", 0),
            "skill_embeddings": type_counts.get("skill", 0),
            "company_embeddings": type_counts.get("company", 0),
            "total_jobs": total_jobs,
            "coverage_pct": (jobs_with_embeddings / total_jobs * 100)
            if total_jobs > 0
            else 0,
            "model_version": model[0] if model else None,
        }

    def delete_embeddings_for_model(self, model_version: str) -> int:
        """
        Delete embeddings for a specific model version.

        Useful when upgrading to a new embedding model.

        Args:
            model_version: Model version to delete

        Returns:
            Number of embeddings deleted
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM embeddings WHERE model_version = ?",
                (model_version,),
            )
            return cursor.rowcount

    def batch_upsert_embeddings(
        self,
        entity_ids: list[str],
        entity_type: str,
        embeddings: np.ndarray,
        model_version: str | None = None,
    ) -> int:
        """
        Batch insert or update embeddings efficiently.

        Args:
            entity_ids: List of entity identifiers
            entity_type: Type of entities
            embeddings: Matrix of shape (n_entities, embedding_dim)
            model_version: Model used

        Returns:
            Number of embeddings upserted
        """
        if len(entity_ids) != len(embeddings):
            raise ValueError("entity_ids and embeddings must have same length")

        model = model_version or "all-MiniLM-L6-v2"
        data = [
            (eid, entity_type, emb.astype(np.float32).tobytes(), model)
            for eid, emb in zip(entity_ids, embeddings)
        ]

        with self._connection() as conn:
            conn.executemany(
                """
                INSERT INTO embeddings (entity_id, entity_type, embedding_blob, model_version, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(entity_id, entity_type) DO UPDATE SET
                    embedding_blob = excluded.embedding_blob,
                    model_version = excluded.model_version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                data,
            )
            return len(data)

    # =========================================================================
    # FTS5 Full-Text Search Methods
    # =========================================================================

    def bm25_search(self, query: str, limit: int = 100) -> list[tuple[str, float]]:
        """
        Full-text search using BM25 ranking.

        Args:
            query: Search query (supports FTS5 query syntax)
            limit: Maximum results to return

        Returns:
            List of (uuid, bm25_score) tuples, sorted by relevance.
            Lower scores = more relevant (BM25 returns negative scores).
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT uuid, bm25(jobs_fts) as score
                FROM jobs_fts
                WHERE jobs_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()

        return [(row[0], row[1]) for row in rows]

    def bm25_search_filtered(
        self, query: str, candidate_uuids: set[str]
    ) -> list[tuple[str, float]]:
        """
        Full-text search restricted to a set of candidate UUIDs.

        Unlike bm25_search which scores globally then filters, this method
        only scores and returns results for the specified candidates. This
        ensures no relevant candidate is missed due to global ranking cutoffs.

        Args:
            query: Search query (supports FTS5 query syntax)
            candidate_uuids: Set of UUIDs to restrict scoring to

        Returns:
            List of (uuid, bm25_score) tuples for matching candidates.
            Lower scores = more relevant (BM25 returns negative scores).
        """
        if not candidate_uuids:
            return []

        with self._connection() as conn:
            # Use a temp table for efficient JOIN with the FTS index.
            # This lets SQLite compute BM25 only for matching candidates
            # rather than ranking the entire corpus first.
            conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _bm25_candidates (uuid TEXT PRIMARY KEY)"
            )
            conn.execute("DELETE FROM _bm25_candidates")
            conn.executemany(
                "INSERT INTO _bm25_candidates (uuid) VALUES (?)",
                [(u,) for u in candidate_uuids],
            )

            rows = conn.execute(
                """
                SELECT f.uuid, bm25(jobs_fts) as score
                FROM jobs_fts f
                INNER JOIN _bm25_candidates c ON c.uuid = f.uuid
                WHERE jobs_fts MATCH ?
                ORDER BY score
                """,
                (query,),
            ).fetchall()

        return [(row[0], row[1]) for row in rows]

    def rebuild_fts_index(self) -> None:
        """
        Rebuild FTS index from jobs table.

        Use this to recover from corruption or after bulk data changes.
        """
        with self._connection() as conn:
            conn.execute("INSERT INTO jobs_fts(jobs_fts) VALUES('rebuild')")
            conn.commit()
        logger.info("FTS5 index rebuilt")

    # =========================================================================
    # Search Analytics Methods
    # =========================================================================

    def log_search(
        self,
        query: str,
        query_type: str,
        result_count: int,
        latency_ms: float,
        cache_hit: bool = False,
        degraded: bool = False,
        filters_used: dict | None = None,
    ) -> None:
        """
        Log a search query for analytics.

        Args:
            query: Search query string
            query_type: 'semantic', 'keyword', or 'hybrid'
            result_count: Number of results returned
            latency_ms: Query execution time in milliseconds
            cache_hit: Whether result came from cache
            degraded: Whether fallback was used
            filters_used: Dict of applied filters
        """
        filters_json = json.dumps(filters_used) if filters_used else None

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO search_analytics
                (query, query_type, result_count, latency_ms, cache_hit, degraded, filters_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (query, query_type, result_count, latency_ms, cache_hit, degraded, filters_json),
            )

    def get_popular_queries(self, days: int = 7, limit: int = 20) -> list[dict]:
        """
        Get most popular search queries in the last N days.

        Args:
            days: Number of days to look back
            limit: Maximum queries to return

        Returns:
            List of dicts with query, count, avg_latency_ms
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT query, COUNT(*) as count, AVG(latency_ms) as avg_latency
                FROM search_analytics
                WHERE searched_at > datetime('now', ?)
                GROUP BY query
                ORDER BY count DESC
                LIMIT ?
                """,
                (f"-{days} days", limit),
            ).fetchall()

        return [
            {"query": r[0], "count": r[1], "avg_latency_ms": r[2]}
            for r in rows
        ]

    def get_search_latency_percentiles(self, days: int = 7) -> dict:
        """
        Get p50, p90, p95, p99 latency statistics.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with percentile values and total count
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT latency_ms FROM search_analytics
                WHERE searched_at > datetime('now', ?)
                ORDER BY latency_ms
                """,
                (f"-{days} days",),
            ).fetchall()

        if not rows:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "count": 0}

        latencies = [r[0] for r in rows]
        n = len(latencies)

        return {
            "p50": latencies[int(n * 0.5)],
            "p90": latencies[int(n * 0.9)],
            "p95": latencies[int(n * 0.95)],
            "p99": latencies[min(int(n * 0.99), n - 1)],
            "count": n,
        }

    def get_analytics_summary(self, days: int = 7) -> dict:
        """
        Get summary of search analytics.

        Args:
            days: Number of days to analyze

        Returns:
            Dict with total searches, cache hit rate, degraded rate, by type
        """
        with self._connection() as conn:
            # Total and by type
            rows = conn.execute(
                """
                SELECT query_type, COUNT(*) as count,
                       SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
                       SUM(CASE WHEN degraded THEN 1 ELSE 0 END) as degraded_count
                FROM search_analytics
                WHERE searched_at > datetime('now', ?)
                GROUP BY query_type
                """,
                (f"-{days} days",),
            ).fetchall()

        by_type = {}
        total = 0
        total_cache_hits = 0
        total_degraded = 0

        for r in rows:
            by_type[r[0]] = {"count": r[1], "cache_hits": r[2], "degraded": r[3]}
            total += r[1]
            total_cache_hits += r[2]
            total_degraded += r[3]

        return {
            "total_searches": total,
            "cache_hit_rate": (total_cache_hits / total * 100) if total > 0 else 0,
            "degraded_rate": (total_degraded / total * 100) if total > 0 else 0,
            "by_type": by_type,
        }

    # =========================================================================
    # Company and Skills Methods (for semantic search features)
    # =========================================================================

    def get_all_companies(self) -> list[str]:
        """
        Get list of all distinct company names.

        Used by embedding generator to enumerate companies for centroid generation.

        Returns:
            Sorted list of company names
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT company_name FROM jobs "
                "WHERE company_name IS NOT NULL AND company_name != '' "
                "ORDER BY company_name"
            ).fetchall()
        return [row[0] for row in rows]

    def get_company_stats(self, company_name: str) -> dict:
        """
        Get statistics for a company.

        Used by similar companies endpoint.

        Args:
            company_name: Company name to look up

        Returns:
            Dict with job_count, avg_salary, top_skills
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as job_count,
                    AVG(salary_annual_min) as avg_salary_min,
                    AVG(salary_annual_max) as avg_salary_max
                FROM jobs
                WHERE company_name = ?
                """,
                (company_name,),
            ).fetchone()

            # Get skills for this company
            skills_rows = conn.execute(
                "SELECT skills FROM jobs WHERE company_name = ? AND skills IS NOT NULL",
                (company_name,),
            ).fetchall()

        # Parse and count skills
        skill_counts: Counter = Counter()
        for r in skills_rows:
            skills = [s.strip() for s in r[0].split(",") if s.strip()]
            skill_counts.update(skills)

        top_skills = [s for s, _ in skill_counts.most_common(10)]

        avg_salary = None
        if row[1] and row[2]:
            avg_salary = int((row[1] + row[2]) / 2)

        return {
            "job_count": row[0],
            "avg_salary": avg_salary,
            "top_skills": top_skills,
        }

    def get_all_unique_skills(self) -> list[str]:
        """
        Extract all unique skills from job postings.

        Returns:
            Sorted list of unique skill names
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT skills FROM jobs WHERE skills IS NOT NULL AND skills != ''"
            ).fetchall()

        skills_set: set[str] = set()
        for row in rows:
            skills = [s.strip() for s in row[0].split(",")]
            skills_set.update(s for s in skills if s)

        return sorted(list(skills_set))

    def get_skill_frequencies(
        self, min_jobs: int = 1, limit: int = 100
    ) -> list[tuple[str, int]]:
        """
        Get skill frequencies for visualization.

        Args:
            min_jobs: Minimum jobs a skill must appear in
            limit: Maximum skills to return

        Returns:
            List of (skill_name, count) tuples, sorted by frequency descending
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT skills FROM jobs WHERE skills IS NOT NULL AND skills != ''"
            ).fetchall()

        skill_counts: Counter = Counter()
        for row in rows:
            skills = [s.strip() for s in row[0].split(",")]
            skill_counts.update(s for s in skills if s)

        filtered = [
            (skill, count)
            for skill, count in skill_counts.items()
            if count >= min_jobs
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)

        return filtered[:limit]

    def get_all_unique_companies(self) -> list[str]:
        """
        Get all unique company names.

        Returns:
            Sorted list of company names
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT company_name FROM jobs
                WHERE company_name IS NOT NULL AND company_name != ''
                """
            ).fetchall()

        return sorted([row[0] for row in rows])

    def get_jobs_without_embeddings(self, limit: int = 1000) -> list[dict]:
        """
        Get jobs that don't have embeddings yet.

        Useful for batch embedding generation.

        Args:
            limit: Maximum jobs to return

        Returns:
            List of job dicts with uuid, title, description, skills
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT j.uuid, j.title, j.description, j.skills, j.company_name
                FROM jobs j
                LEFT JOIN embeddings e ON j.uuid = e.entity_id AND e.entity_type = 'job'
                WHERE e.id IS NULL
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]
