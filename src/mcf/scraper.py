"""
Job scraper orchestration for MyCareersFuture.

Coordinates the API client and storage layer with:
- Concurrent job fetching with semaphore control
- Checkpoint-based resume after interruption
- Graceful shutdown on Ctrl+C
- Progress tracking with callbacks
"""

import asyncio
import logging
import signal
from datetime import datetime
from typing import Callable, Optional

from rich.progress import Progress, TaskID

from .api_client import MCFClient, MCFAPIError, MCFRateLimitError
from .models import Job, Checkpoint
from .storage import JobStorage, SQLiteStorage

logger = logging.getLogger(__name__)


class MCFScraper:
    """
    High-level scraper that orchestrates job collection.

    Features:
    - Concurrent fetching with configurable parallelism
    - Automatic checkpointing every N jobs
    - Resume from previous incomplete runs
    - Graceful shutdown preserving progress
    - Deduplication across multiple search queries

    Example:
        scraper = MCFScraper()
        jobs = await scraper.scrape("data scientist", max_jobs=1000)
        scraper.save("data_scientist")
    """

    def __init__(
        self,
        output_dir: str = "data",
        checkpoint_interval: int = 100,
        concurrency: int = 5,
        requests_per_second: float = 2.0,
        use_sqlite: bool = True,
        db_path: str = "data/mcf_jobs.db",
    ):
        """
        Initialize the scraper.

        Args:
            output_dir: Directory for output files
            checkpoint_interval: Save checkpoint every N jobs
            concurrency: Maximum concurrent API requests
            requests_per_second: Rate limit for API requests
            use_sqlite: Use SQLite storage with history tracking (default: True)
            db_path: Path to SQLite database file
        """
        self.use_sqlite = use_sqlite
        if use_sqlite:
            self.storage = SQLiteStorage(db_path=db_path, output_dir=output_dir)
        else:
            self.storage = JobStorage(output_dir=output_dir)

        self.checkpoint_interval = checkpoint_interval
        self.concurrency = concurrency
        self.requests_per_second = requests_per_second

        self._shutdown_event = asyncio.Event()
        self._current_query: Optional[str] = None
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        def signal_handler():
            logger.warning("Shutdown requested, saving progress...")
            self._shutdown_event.set()

        try:
            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    async def scrape(
        self,
        query: str,
        max_jobs: Optional[int] = None,
        resume: bool = True,
        progress: Optional[Progress] = None,
    ) -> list[Job]:
        """
        Scrape jobs matching a search query.

        Args:
            query: Search query (e.g., "data scientist")
            max_jobs: Maximum jobs to fetch (None for all)
            resume: Whether to resume from checkpoint if available
            progress: Rich Progress instance for progress tracking

        Returns:
            List of scraped Job objects
        """
        self._current_query = query
        self._progress = progress
        self._shutdown_event.clear()
        self._setup_signal_handlers()

        # Check for existing checkpoint
        checkpoint = None
        start_offset = 0
        existing_uuids: set[str] = set()

        if resume:
            checkpoint = self.storage.load_checkpoint(query)
            if checkpoint:
                start_offset = checkpoint.current_offset
                existing_uuids = set(checkpoint.job_uuids)
                logger.info(
                    f"Resuming from checkpoint: {checkpoint.fetched_count} jobs, "
                    f"offset {start_offset}"
                )

        async with MCFClient(requests_per_second=self.requests_per_second) as client:
            # Get total job count for progress tracking
            total = await client.get_total_jobs(query)
            logger.info(f"Found {total} jobs matching '{query}'")

            if max_jobs:
                total = min(total, max_jobs)

            # Set up progress bar
            if progress:
                self._task_id = progress.add_task(
                    f"Scraping '{query}'",
                    total=total,
                    completed=len(existing_uuids),
                )

            # Start session for SQLite storage (if not resuming)
            if self.use_sqlite and not checkpoint:
                self.storage.start_session(query, total)

            # Create checkpoint object
            checkpoint = Checkpoint(
                search_query=query,
                total_jobs=total,
                fetched_count=len(existing_uuids),
                current_offset=start_offset,
                job_uuids=list(existing_uuids),
            )

            # Fetch jobs with batching
            batch_size = 100
            offset = start_offset
            fetched_since_checkpoint = 0

            while offset < total and not self._shutdown_event.is_set():
                remaining = total - offset
                current_batch_size = min(batch_size, remaining)

                try:
                    response = await client.search_jobs(
                        query,
                        limit=current_batch_size,
                        offset=offset,
                    )
                except MCFRateLimitError:
                    backoff = 1.0 / self.requests_per_second + 5.0
                    logger.warning(
                        f"Rate limited at offset {offset}, "
                        f"backing off {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    continue
                except MCFAPIError as e:
                    logger.error(f"API error at offset {offset}: {e}")
                    break

                if not response.results:
                    logger.warning(f"Empty response at offset {offset}")
                    break

                # Add jobs to storage (deduplication happens here)
                new_count = 0
                for job in response.results:
                    if job.uuid not in existing_uuids:
                        if self.storage.add_job(job):
                            new_count += 1
                            checkpoint.job_uuids.append(job.uuid)
                            existing_uuids.add(job.uuid)

                fetched_since_checkpoint += len(response.results)
                offset += len(response.results)

                # Update progress
                if progress and self._task_id is not None:
                    progress.update(self._task_id, completed=len(existing_uuids))

                # Save checkpoint periodically
                if fetched_since_checkpoint >= self.checkpoint_interval:
                    checkpoint.current_offset = offset
                    checkpoint.fetched_count = len(existing_uuids)
                    self.storage.save_checkpoint(checkpoint)
                    fetched_since_checkpoint = 0
                    logger.debug(f"Checkpoint saved at offset {offset}")

                # Check for max_jobs limit
                if max_jobs and len(existing_uuids) >= max_jobs:
                    logger.info(f"Reached max_jobs limit ({max_jobs})")
                    break

            # Final checkpoint save
            checkpoint.current_offset = offset
            checkpoint.fetched_count = len(existing_uuids)

            if self._shutdown_event.is_set():
                # Save checkpoint on shutdown for resume
                self.storage.save_checkpoint(checkpoint)
                logger.info(f"Progress saved. Resume with same query to continue.")
            else:
                # Clear checkpoint on successful completion
                self.storage.clear_checkpoint(query)
                logger.info(f"Completed: {len(existing_uuids)} jobs scraped")

        return self.storage.jobs

    async def scrape_multiple(
        self,
        queries: list[str],
        max_jobs_per_query: Optional[int] = None,
        progress: Optional[Progress] = None,
    ) -> list[Job]:
        """
        Scrape jobs for multiple search queries.

        Jobs are deduplicated across all queries.

        Args:
            queries: List of search queries
            max_jobs_per_query: Max jobs per query (None for all)
            progress: Rich Progress instance

        Returns:
            List of all unique jobs across queries
        """
        for query in queries:
            if self._shutdown_event.is_set():
                break

            logger.info(f"Starting scrape for '{query}'")
            await self.scrape(
                query,
                max_jobs=max_jobs_per_query,
                progress=progress,
            )

        logger.info(f"Total unique jobs: {self.storage.job_count}")
        return self.storage.jobs

    def save(
        self,
        name: str,
        format: str = "csv",
        include_date: bool = True,
    ) -> str:
        """
        Save scraped jobs to file.

        Args:
            name: Base name for output file
            format: Output format ("csv" or "json")
            include_date: Whether to include date in filename

        Returns:
            Path to saved file
        """
        if format == "csv":
            return str(self.storage.save_csv(name, include_date=include_date))
        elif format == "json":
            return str(self.storage.save_json(name, include_date=include_date))
        else:
            raise ValueError(f"Unsupported format: {format}")

    @property
    def job_count(self) -> int:
        """Number of unique jobs collected."""
        return self.storage.job_count

    def get_dataframe(self):
        """Get jobs as a pandas DataFrame."""
        return self.storage.to_dataframe()
