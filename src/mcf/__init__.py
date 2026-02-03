"""
MCF Scraper - API-based job scraper for MyCareersFuture.sg

This package provides a fast, reliable way to scrape job listings
using the public MyCareersFuture API instead of browser automation.

Features:
- Fast async API-based scraping
- SQLite storage with automatic deduplication
- History tracking when jobs are updated
- Resume support for interrupted scrapes
- Per-ID attempt tracking for gap detection
- Adaptive rate limiting
- Daemon mode for long-running scrapes
- CSV/JSON export
"""

from .models import Job, JobSearchResponse
from .api_client import MCFClient
from .scraper import MCFScraper
from .storage import JobStorage, SQLiteStorage
from .database import MCFDatabase
from .migration import MCFMigrator, MigrationStats, LegacyJobParser
from .historical_scraper import HistoricalScraper, ScrapeProgress, YEAR_ESTIMATES
from .batch_logger import BatchLogger
from .adaptive_rate import AdaptiveRateLimiter, RateState
from .daemon import ScraperDaemon, DaemonError, DaemonAlreadyRunning, DaemonNotRunning

__all__ = [
    # Core models
    "Job",
    "JobSearchResponse",
    # API client
    "MCFClient",
    # Scrapers
    "MCFScraper",
    "HistoricalScraper",
    "ScrapeProgress",
    "YEAR_ESTIMATES",
    # Storage
    "JobStorage",
    "SQLiteStorage",
    "MCFDatabase",
    # Migration
    "MCFMigrator",
    "MigrationStats",
    "LegacyJobParser",
    # Robust pipeline components
    "BatchLogger",
    "AdaptiveRateLimiter",
    "RateState",
    "ScraperDaemon",
    "DaemonError",
    "DaemonAlreadyRunning",
    "DaemonNotRunning",
]
__version__ = "1.4.0"
