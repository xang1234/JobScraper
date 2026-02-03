"""
Async HTTP client for the MyCareersFuture API.

Provides robust access to the MCF public API with:
- Automatic retry with exponential backoff
- Rate limiting to be respectful of the API
- Proper error handling and logging
"""

import asyncio
import logging
from typing import Optional
from urllib.parse import urlencode

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .models import Job, JobSearchResponse

logger = logging.getLogger(__name__)


class MCFAPIError(Exception):
    """Base exception for MCF API errors."""
    pass


class MCFRateLimitError(MCFAPIError):
    """Raised when rate limited by the API."""
    pass


class MCFNotFoundError(MCFAPIError):
    """Raised when a resource is not found."""
    pass


class MCFClient:
    """
    Async client for the MyCareersFuture API.

    Features:
    - Automatic retry with exponential backoff (3 attempts)
    - Rate limiting (configurable requests per second)
    - Connection pooling via httpx
    - Proper error classification

    Example:
        async with MCFClient() as client:
            response = await client.search_jobs("data scientist", limit=20)
            for job in response.results:
                print(job.title)
    """

    BASE_URL = "https://api.mycareersfuture.gov.sg/v2"
    DEFAULT_HEADERS = {
        "Accept": "application/json",
        "User-Agent": "MCF-Scraper/1.0 (Job Market Research)",
    }

    def __init__(
        self,
        requests_per_second: float = 1.0,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize the MCF API client.

        Args:
            requests_per_second: Rate limit for API requests (default 1/sec)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.requests_per_second = requests_per_second
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request_time: float = 0
        self._rate_limit_lock = asyncio.Lock()

    async def __aenter__(self) -> "MCFClient":
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=self.DEFAULT_HEADERS,
            timeout=self.timeout,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        async with self._rate_limit_lock:
            now = asyncio.get_event_loop().time()
            min_interval = 1.0 / self.requests_per_second
            elapsed = now - self._last_request_time

            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            self._last_request_time = asyncio.get_event_loop().time()

    def _handle_response(self, response: httpx.Response) -> dict:
        """
        Handle API response and raise appropriate exceptions.

        Args:
            response: The HTTP response

        Returns:
            Parsed JSON data

        Raises:
            MCFRateLimitError: If rate limited (429)
            MCFNotFoundError: If resource not found (404)
            MCFAPIError: For other HTTP errors
        """
        if response.status_code == 429:
            raise MCFRateLimitError("Rate limited by MCF API")

        if response.status_code == 404:
            raise MCFNotFoundError(f"Resource not found: {response.url}")

        if response.status_code >= 400:
            raise MCFAPIError(
                f"API error {response.status_code}: {response.text[:200]}"
            )

        return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, MCFRateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """
        Make an API request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            **kwargs: Additional arguments for httpx

        Returns:
            Parsed JSON response
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        await self._rate_limit()

        logger.debug(f"Request: {method} {path}")
        response = await self._client.request(method, path, **kwargs)
        return self._handle_response(response)

    async def search_jobs(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "new_posting_date",
        employment_types: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
    ) -> JobSearchResponse:
        """
        Search for jobs using the MCF API.

        Args:
            query: Search query string (e.g., "data scientist")
            limit: Number of results per page (max 100)
            offset: Starting offset for pagination
            sort_by: Sort order ("new_posting_date", "relevancy", "salary")
            employment_types: Filter by employment type
            categories: Filter by job category

        Returns:
            JobSearchResponse with results and pagination info
        """
        params = {
            "search": query,
            "limit": min(limit, 100),  # API max is 100
            "page": offset // limit if limit > 0 else 0,
            "sortBy": sort_by,
        }

        if employment_types:
            params["employmentTypes"] = ",".join(employment_types)

        if categories:
            params["categories"] = ",".join(categories)

        data = await self._request("GET", "/jobs", params=params)

        # Parse results into Job models
        jobs = []
        for job_data in data.get("results", []):
            try:
                jobs.append(Job.model_validate(job_data))
            except Exception as e:
                logger.warning(f"Failed to parse job: {e}")
                continue

        return JobSearchResponse(
            results=jobs,
            total=data.get("total", 0),
            offset=offset,
            limit=limit,
        )

    async def get_job(self, uuid: str) -> Job:
        """
        Get detailed information for a specific job.

        Args:
            uuid: The job's unique identifier

        Returns:
            Job with full details

        Raises:
            MCFNotFoundError: If job doesn't exist
        """
        data = await self._request("GET", f"/jobs/{uuid}")
        return Job.model_validate(data)

    async def get_total_jobs(self, query: str) -> int:
        """
        Get total number of jobs matching a query.

        This is useful for progress tracking and pagination planning.

        Args:
            query: Search query string

        Returns:
            Total number of matching jobs
        """
        response = await self.search_jobs(query, limit=1, offset=0)
        return response.total

    async def iter_jobs(
        self,
        query: str,
        max_jobs: Optional[int] = None,
        start_offset: int = 0,
        batch_size: int = 100,
    ):
        """
        Async generator that yields jobs page by page.

        This is memory-efficient for large result sets as it
        doesn't load all jobs into memory at once.

        Args:
            query: Search query string
            max_jobs: Maximum number of jobs to fetch (None for all)
            start_offset: Starting offset for pagination
            batch_size: Number of jobs per API request

        Yields:
            Job objects one at a time
        """
        offset = start_offset
        fetched = 0

        while True:
            response = await self.search_jobs(
                query,
                limit=batch_size,
                offset=offset,
            )

            if not response.results:
                break

            for job in response.results:
                yield job
                fetched += 1

                if max_jobs and fetched >= max_jobs:
                    return

            offset += len(response.results)

            # Stop if we've fetched all available jobs
            if offset >= response.total:
                break
