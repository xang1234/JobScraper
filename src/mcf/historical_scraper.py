"""
Historical job scraper for MyCareersFuture.

Enumerates and fetches all historical MCF jobs by generating jobPostId values
and converting them to UUIDs via MD5 hash. This allows scraping the complete
job database from 2019 to present.

UUID Discovery:
- UUID = MD5(jobPostId) where jobPostId format is MCF-{YEAR}-{7-digit sequence}
- API endpoint /v2/jobs/{uuid} returns full job data even for closed/historical jobs
- Job IDs are sequential per year (starting at 0000001)

Enhanced Features:
- Per-ID attempt tracking via BatchLogger for gap detection
- Adaptive rate limiting that backs off on 429s and recovers slowly
- Daemon mode support for long-running background operation
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, Awaitable

from tenacity import RetryError

from .api_client import MCFClient, MCFNotFoundError, MCFRateLimitError, MCFAPIError
from .adaptive_rate import AdaptiveRateLimiter
from .batch_logger import BatchLogger
from .database import MCFDatabase
from .models import Job

logger = logging.getLogger(__name__)

# Estimated max sequences per year (conservative estimates)
YEAR_ESTIMATES = {
    2019: 50_000,
    2020: 350_000,
    2021: 700_000,
    2022: 1_000_000,
    2023: 1_000_000,
    2024: 1_450_000,
    2025: 1_500_000,
    2026: 250_000,  # Growing
}

# Stop scanning after this many consecutive not-found responses
DEFAULT_NOT_FOUND_THRESHOLD = 1000


@dataclass
class ScrapeProgress:
    """Progress information for a scraping session."""
    year: int
    current_seq: int
    jobs_found: int
    jobs_not_found: int
    consecutive_not_found: int
    start_seq: int
    end_seq: Optional[int]

    @property
    def total_processed(self) -> int:
        return self.jobs_found + self.jobs_not_found

    @property
    def success_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.jobs_found / self.total_processed * 100


class HistoricalScraper:
    """
    Enumerate and fetch all historical MCF jobs by jobPostId.

    This scraper generates possible job IDs (MCF-YYYY-NNNNNNN format),
    converts them to UUIDs via MD5 hash, and fetches each job from the API.

    Features:
    - Resume support via SQLite session tracking
    - Adaptive rate limiting with backoff on 429 errors
    - Per-ID attempt tracking for gap detection and retry
    - Skips already-fetched jobs (deduplication)
    - Detects end-of-year via consecutive not-found threshold

    Example:
        async with HistoricalScraper("data/mcf_jobs.db") as scraper:
            await scraper.scrape_year(2023)
    """

    def __init__(
        self,
        db_path: str = "data/mcf_jobs.db",
        requests_per_second: float = 2.0,
        not_found_threshold: int = DEFAULT_NOT_FOUND_THRESHOLD,
        batch_size: int = 50,
        min_rps: float = 0.5,
        max_rps: float = 5.0,
    ):
        """
        Initialize the historical scraper.

        Args:
            db_path: Path to SQLite database
            requests_per_second: Initial rate limit for API requests
            not_found_threshold: Stop after this many consecutive not-found responses
            batch_size: Number of attempts to buffer before flushing to DB
            min_rps: Minimum requests per second (during heavy rate limiting)
            max_rps: Maximum requests per second (after recovery)
        """
        self.db = MCFDatabase(db_path)
        self.initial_rps = requests_per_second
        self.not_found_threshold = not_found_threshold
        self._client: Optional[MCFClient] = None
        self._existing_uuids: set[str] = set()

        # New components for robust operation
        self.batch_logger = BatchLogger(self.db, batch_size=batch_size)
        self.rate_limiter = AdaptiveRateLimiter(
            initial_rps=requests_per_second,
            min_rps=min_rps,
            max_rps=max_rps,
        )

    async def __aenter__(self) -> "HistoricalScraper":
        """Async context manager entry."""
        self._client = MCFClient(requests_per_second=self.initial_rps)
        await self._client.__aenter__()
        # Load existing UUIDs for deduplication
        self._existing_uuids = self.db.get_all_uuids()
        logger.info(f"Loaded {len(self._existing_uuids):,} existing job UUIDs")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # Flush any pending batch logger entries
        self.batch_logger.flush()

        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    @staticmethod
    def job_id_to_uuid(job_id: str) -> str:
        """
        Convert MCF-YYYY-NNNNNNN to UUID (MD5 hash).

        Args:
            job_id: Job ID in format MCF-2023-0000001

        Returns:
            32-character hex string (MD5 hash)
        """
        return hashlib.md5(job_id.encode()).hexdigest()

    @staticmethod
    def format_job_id(year: int, sequence: int) -> str:
        """
        Format a job ID from year and sequence number.

        Args:
            year: Year (e.g., 2023)
            sequence: Sequence number (1-9999999)

        Returns:
            Job ID in format MCF-2023-0000001
        """
        return f"MCF-{year}-{sequence:07d}"

    @staticmethod
    def parse_job_id(job_id: str) -> tuple[int, int]:
        """
        Parse a job ID into year and sequence.

        Args:
            job_id: Job ID in format MCF-2023-0000001

        Returns:
            Tuple of (year, sequence)
        """
        parts = job_id.split("-")
        if len(parts) != 3 or parts[0] != "MCF":
            raise ValueError(f"Invalid job ID format: {job_id}")
        return int(parts[1]), int(parts[2])

    async def fetch_job(self, year: int, sequence: int) -> Optional[Job]:
        """
        Fetch a single job by year and sequence.

        Args:
            year: Year of the job
            sequence: Sequence number

        Returns:
            Job if found, None if not found

        Raises:
            MCFRateLimitError: If rate limited (caller should back off)
            MCFAPIError: For other API errors
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        job_id = self.format_job_id(year, sequence)
        uuid = self.job_id_to_uuid(job_id)

        # Skip if already in database
        if uuid in self._existing_uuids:
            logger.debug(f"Skipping existing job: {job_id}")
            return None

        try:
            job = await self._client.get_job(uuid)
            return job
        except MCFNotFoundError:
            return None

    async def scrape_year(
        self,
        year: int,
        start_seq: int = 1,
        end_seq: Optional[int] = None,
        resume: bool = True,
        progress_callback: Optional[Callable[[ScrapeProgress], Awaitable[None]]] = None,
        dry_run: bool = False,
    ) -> ScrapeProgress:
        """
        Scrape all jobs for a given year.

        Args:
            year: Year to scrape (e.g., 2023)
            start_seq: Starting sequence number (default: 1)
            end_seq: Ending sequence number (default: estimated max for year)
            resume: Whether to resume from previous session
            progress_callback: Async callback for progress updates
            dry_run: If True, only preview without fetching

        Returns:
            Final scrape progress
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        # Determine end sequence
        if end_seq is None:
            end_seq = YEAR_ESTIMATES.get(year, 1_000_000)

        # Check for existing session to resume
        session_id: Optional[int] = None
        jobs_found = 0
        jobs_not_found = 0
        consecutive_not_found = 0

        if resume:
            existing_session = self.db.get_incomplete_historical_session(year)
            if existing_session:
                session_id = existing_session["id"]
                start_seq = existing_session["current_seq"] + 1
                jobs_found = existing_session["jobs_found"]
                jobs_not_found = existing_session["jobs_not_found"]
                consecutive_not_found = existing_session["consecutive_not_found"]
                logger.info(
                    f"Resuming year {year} from sequence {start_seq:,} "
                    f"({jobs_found:,} found, {jobs_not_found:,} not found)"
                )

        # Create new session if needed
        if session_id is None and not dry_run:
            session_id = self.db.create_historical_session(year, start_seq, end_seq)
            logger.info(f"Created new session {session_id} for year {year}")

        current_seq = start_seq
        checkpoint_interval = 100  # Save progress every N jobs

        logger.info(
            f"Scraping year {year}: sequences {start_seq:,} to {end_seq:,}"
            f"{' (DRY RUN)' if dry_run else ''}"
        )
        logger.info(f"Rate limiter: {self.rate_limiter.current_rps:.2f} req/sec")

        try:
            while current_seq <= end_seq:
                # Check for early termination
                if consecutive_not_found >= self.not_found_threshold:
                    logger.info(
                        f"Year {year}: {consecutive_not_found} consecutive not-found, "
                        "assuming end of sequence"
                    )
                    break

                if dry_run:
                    # In dry run, just count without fetching
                    job_id = self.format_job_id(year, current_seq)
                    uuid = self.job_id_to_uuid(job_id)
                    if uuid in self._existing_uuids:
                        jobs_found += 1
                        self.batch_logger.log(year, current_seq, 'skipped')
                    else:
                        jobs_not_found += 1
                        self.batch_logger.log(year, current_seq, 'not_found')
                    current_seq += 1
                    continue

                try:
                    job = await self.fetch_job(year, current_seq)

                    if job:
                        # Save to database
                        is_new, was_updated = self.db.upsert_job(job)
                        self._existing_uuids.add(job.uuid)
                        jobs_found += 1
                        consecutive_not_found = 0

                        # Log successful fetch
                        self.batch_logger.log(year, current_seq, 'found')
                        self.rate_limiter.on_success()

                        # Update client rate if changed significantly
                        if self._client:
                            self._client.requests_per_second = self.rate_limiter.current_rps

                        if is_new:
                            logger.debug(f"New job: {job.title[:50]} ({job.company_name})")
                    else:
                        jobs_not_found += 1
                        # Check if it was skipped (already exists) vs truly not found
                        uuid = self.job_id_to_uuid(self.format_job_id(year, current_seq))
                        if uuid not in self._existing_uuids:
                            consecutive_not_found += 1
                            self.batch_logger.log(year, current_seq, 'not_found')
                        else:
                            consecutive_not_found = 0
                            self.batch_logger.log(year, current_seq, 'skipped')
                        self.rate_limiter.on_success()

                except MCFRateLimitError:
                    # Use adaptive rate limiter for backoff
                    new_rps = self.rate_limiter.on_rate_limited()
                    if self._client:
                        self._client.requests_per_second = new_rps

                    # Wait based on new rate (inverse of RPS)
                    backoff_delay = 1.0 / new_rps + 1.0  # Extra 1s buffer
                    logger.warning(
                        f"Rate limited at seq {current_seq}, "
                        f"backing off {backoff_delay:.1f}s, new rate: {new_rps:.2f} req/sec"
                    )
                    await asyncio.sleep(backoff_delay)
                    continue  # Don't increment sequence, retry

                except MCFAPIError as e:
                    logger.error(f"API error at {year}-{current_seq}: {e}")
                    self.batch_logger.log(year, current_seq, 'error', str(e))
                    self.rate_limiter.on_error()
                    jobs_not_found += 1
                    consecutive_not_found += 1

                except RetryError as e:
                    # Tenacity exhausted retries - check if underlying cause was rate limit
                    cause = e.last_attempt.exception() if e.last_attempt else None
                    if isinstance(cause, MCFRateLimitError):
                        new_rps = self.rate_limiter.on_rate_limited()
                        if self._client:
                            self._client.requests_per_second = new_rps
                        backoff_delay = 1.0 / new_rps + 5.0  # Longer backoff after retry exhaustion
                        logger.warning(
                            f"Retries exhausted due to rate limiting at seq {current_seq}, "
                            f"backing off {backoff_delay:.1f}s, new rate: {new_rps:.2f} req/sec"
                        )
                        await asyncio.sleep(backoff_delay)
                        continue  # Retry same sequence
                    else:
                        # Other retry error - log and continue
                        logger.error(f"Retry exhausted at {year}-{current_seq}: {e}")
                        self.batch_logger.log(year, current_seq, 'error', str(e))
                        self.rate_limiter.on_error()
                        jobs_not_found += 1
                        consecutive_not_found += 1

                except Exception as e:
                    # Catch-all for unexpected errors - log and continue
                    logger.exception(f"Unexpected error at {year}-{current_seq}: {e}")
                    self.batch_logger.log(year, current_seq, 'error', str(e))
                    self.rate_limiter.on_error()
                    jobs_not_found += 1
                    consecutive_not_found += 1

                # Update progress
                current_seq += 1

                # Checkpoint periodically
                if session_id and (current_seq - start_seq) % checkpoint_interval == 0:
                    self.db.update_historical_progress(
                        session_id,
                        current_seq,
                        jobs_found,
                        jobs_not_found,
                        consecutive_not_found,
                    )

                    # Progress callback
                    if progress_callback:
                        progress = ScrapeProgress(
                            year=year,
                            current_seq=current_seq,
                            jobs_found=jobs_found,
                            jobs_not_found=jobs_not_found,
                            consecutive_not_found=consecutive_not_found,
                            start_seq=start_seq,
                            end_seq=end_seq,
                        )
                        await progress_callback(progress)

        finally:
            # Flush batch logger
            self.batch_logger.flush()

            # Final progress update
            if session_id:
                self.db.update_historical_progress(
                    session_id,
                    current_seq,
                    jobs_found,
                    jobs_not_found,
                    consecutive_not_found,
                )

        # Mark completed if we finished normally
        if session_id and current_seq > end_seq:
            self.db.complete_historical_session(session_id)
            logger.info(f"Completed year {year}")

        return ScrapeProgress(
            year=year,
            current_seq=current_seq,
            jobs_found=jobs_found,
            jobs_not_found=jobs_not_found,
            consecutive_not_found=consecutive_not_found,
            start_seq=start_seq,
            end_seq=end_seq,
        )

    async def scrape_range(
        self,
        start_id: str,
        end_id: str,
        progress_callback: Optional[Callable[[ScrapeProgress], Awaitable[None]]] = None,
        dry_run: bool = False,
    ) -> ScrapeProgress:
        """
        Scrape a specific range of job IDs.

        Args:
            start_id: Starting job ID (e.g., MCF-2023-0500000)
            end_id: Ending job ID (e.g., MCF-2023-0600000)
            progress_callback: Async callback for progress updates
            dry_run: If True, only preview without fetching

        Returns:
            Scrape progress
        """
        start_year, start_seq = self.parse_job_id(start_id)
        end_year, end_seq = self.parse_job_id(end_id)

        if start_year != end_year:
            raise ValueError("Start and end IDs must be from the same year")

        return await self.scrape_year(
            start_year,
            start_seq=start_seq,
            end_seq=end_seq,
            resume=False,  # Don't resume for explicit range
            progress_callback=progress_callback,
            dry_run=dry_run,
        )

    async def scrape_all_years(
        self,
        years: Optional[list[int]] = None,
        resume: bool = True,
        progress_callback: Optional[Callable[[ScrapeProgress], Awaitable[None]]] = None,
        dry_run: bool = False,
    ) -> dict[int, ScrapeProgress]:
        """
        Scrape all years (2019-present).

        Args:
            years: List of years to scrape (default: all known years)
            resume: Whether to resume from previous sessions
            progress_callback: Async callback for progress updates
            dry_run: If True, only preview without fetching

        Returns:
            Dict mapping year to final progress
        """
        if years is None:
            years = sorted(YEAR_ESTIMATES.keys())

        results = {}
        for year in years:
            logger.info(f"Starting year {year}")
            results[year] = await self.scrape_year(
                year,
                resume=resume,
                progress_callback=progress_callback,
                dry_run=dry_run,
            )

        return results

    async def find_year_bounds(self, year: int) -> tuple[int, int]:
        """
        Use binary search to find the valid sequence range for a year.

        This helps avoid wasted requests on non-existent IDs.

        Args:
            year: Year to find bounds for

        Returns:
            Tuple of (min_seq, max_seq) that exist
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        # Find minimum (usually 1, but check)
        min_seq = 1

        # Binary search for maximum
        low = 1
        high = YEAR_ESTIMATES.get(year, 1_000_000) * 2  # Search beyond estimate

        while low < high:
            mid = (low + high + 1) // 2
            uuid = self.job_id_to_uuid(self.format_job_id(year, mid))

            try:
                await self._client.get_job(uuid)
                low = mid  # Found, search higher
            except MCFNotFoundError:
                high = mid - 1  # Not found, search lower
            except MCFRateLimitError:
                # Back off and retry
                await asyncio.sleep(5.0)
                continue

        return (min_seq, low)

    async def retry_gaps(
        self,
        year: int,
        progress_callback: Optional[Callable[[ScrapeProgress], Awaitable[None]]] = None,
    ) -> ScrapeProgress:
        """
        Retry fetching jobs for missing/failed sequences in a year.

        Finds gaps in fetch_attempts and errors, then retries each.

        Args:
            year: Year to retry gaps for
            progress_callback: Async callback for progress updates

        Returns:
            Scrape progress with retry results
        """
        if not self._client:
            raise RuntimeError("Scraper not initialized. Use 'async with' context.")

        # Get gaps and failed attempts
        gaps = self.db.get_missing_sequences(year)
        failed = self.db.get_failed_attempts(year)

        # Build list of sequences to retry
        sequences_to_retry: list[int] = []

        for start, end in gaps:
            sequences_to_retry.extend(range(start, end + 1))

        for attempt in failed:
            if attempt["sequence"] not in sequences_to_retry:
                sequences_to_retry.append(attempt["sequence"])

        sequences_to_retry.sort()

        if not sequences_to_retry:
            logger.info(f"No gaps or failed attempts to retry for year {year}")
            return ScrapeProgress(
                year=year,
                current_seq=0,
                jobs_found=0,
                jobs_not_found=0,
                consecutive_not_found=0,
                start_seq=0,
                end_seq=0,
            )

        logger.info(
            f"Retrying {len(sequences_to_retry):,} sequences for year {year} "
            f"(gaps: {len(gaps)}, failed: {len(failed)})"
        )

        jobs_found = 0
        jobs_not_found = 0

        for i, seq in enumerate(sequences_to_retry):
            try:
                job = await self.fetch_job(year, seq)

                if job:
                    is_new, _ = self.db.upsert_job(job)
                    self._existing_uuids.add(job.uuid)
                    jobs_found += 1
                    self.batch_logger.log(year, seq, 'found')
                    self.rate_limiter.on_success()

                    if is_new:
                        logger.debug(f"Recovered job: {job.title[:50]}")
                else:
                    uuid = self.job_id_to_uuid(self.format_job_id(year, seq))
                    if uuid in self._existing_uuids:
                        self.batch_logger.log(year, seq, 'skipped')
                    else:
                        self.batch_logger.log(year, seq, 'not_found')
                        jobs_not_found += 1
                    self.rate_limiter.on_success()

            except MCFRateLimitError:
                new_rps = self.rate_limiter.on_rate_limited()
                if self._client:
                    self._client.requests_per_second = new_rps
                await asyncio.sleep(1.0 / new_rps + 1.0)
                # Log as error for retry later
                self.batch_logger.log(year, seq, 'error', 'rate_limited')

            except MCFAPIError as e:
                self.batch_logger.log(year, seq, 'error', str(e))
                self.rate_limiter.on_error()
                jobs_not_found += 1

            # Progress callback every 100 sequences
            if progress_callback and (i + 1) % 100 == 0:
                progress = ScrapeProgress(
                    year=year,
                    current_seq=seq,
                    jobs_found=jobs_found,
                    jobs_not_found=jobs_not_found,
                    consecutive_not_found=0,
                    start_seq=sequences_to_retry[0],
                    end_seq=sequences_to_retry[-1],
                )
                await progress_callback(progress)

        # Final flush
        self.batch_logger.flush()

        logger.info(
            f"Gap retry complete for year {year}: "
            f"{jobs_found:,} recovered, {jobs_not_found:,} still missing"
        )

        return ScrapeProgress(
            year=year,
            current_seq=sequences_to_retry[-1] if sequences_to_retry else 0,
            jobs_found=jobs_found,
            jobs_not_found=jobs_not_found,
            consecutive_not_found=0,
            start_seq=sequences_to_retry[0] if sequences_to_retry else 0,
            end_seq=sequences_to_retry[-1] if sequences_to_retry else 0,
        )

    async def retry_all_gaps(
        self,
        years: Optional[list[int]] = None,
        progress_callback: Optional[Callable[[ScrapeProgress], Awaitable[None]]] = None,
    ) -> dict[int, ScrapeProgress]:
        """
        Retry gaps for all years.

        Args:
            years: List of years to retry (default: all years with attempts)
            progress_callback: Async callback for progress updates

        Returns:
            Dict mapping year to retry results
        """
        if years is None:
            # Get years from attempt stats
            stats = self.db.get_all_attempt_stats()
            years = sorted(stats.keys())

        results = {}
        for year in years:
            results[year] = await self.retry_gaps(year, progress_callback)

        return results
