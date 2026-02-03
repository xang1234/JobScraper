# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Job market data collection system that scrapes job listings from MyCareersFuture.sg and LinkedIn. Focuses on tech roles (Data Science, Machine Learning, Data Engineering, Quantitative Analysis) for job market analysis.

## Running the Scrapers

### MCF API Scraper (Recommended)

The API-based scraper is fast and reliable, storing data in SQLite with automatic deduplication and history tracking.

```bash
# Install dependencies
poetry install

# Quick preview (no file output)
python -m src.cli preview "data scientist"

# Scrape jobs (stored in SQLite + CSV export)
python -m src.cli scrape "data scientist"
python -m src.cli scrape "machine learning" --max-jobs 500

# Scrape with JSON output
python -m src.cli scrape "data engineer" --format json

# Scrape multiple queries with deduplication
python -m src.cli scrape-multi "data scientist" "machine learning" "data engineer"
```

**Scrape Options:**
- `--max-jobs, -n` - Limit number of jobs to scrape
- `--output, -o` - Output directory (default: `data/`)
- `--format, -f` - Output format: `csv` or `json`
- `--no-resume` - Don't resume from checkpoint
- `--rate-limit, -r` - Requests per second (default: 2.0)
- `--verbose, -v` - Enable debug logging

### Database Query Commands

Query and analyze scraped jobs from the SQLite database:

```bash
# List jobs with filters
python -m src.cli list --limit 20
python -m src.cli list --company "Google" --salary-min 8000
python -m src.cli list --employment-type "Permanent"

# Search by keyword (searches title, description, skills)
python -m src.cli search "machine learning"
python -m src.cli search "Python" --limit 50

# Show database statistics
python -m src.cli stats

# Export filtered jobs to CSV
python -m src.cli export jobs.csv
python -m src.cli export high_salary.csv --salary-min 10000
python -m src.cli export google_jobs.csv --company Google

# View job change history
python -m src.cli history <uuid>

# Check scrape session status
python -m src.cli db-status
python -m src.cli status  # Legacy checkpoint status
```

**Query Options:**
- `--limit, -n` - Number of results to show
- `--company, -c` - Filter by company name (partial match)
- `--salary-min` - Minimum salary filter
- `--salary-max` - Maximum salary filter
- `--employment-type, -e` - Filter by employment type
- `--db` - Database path (default: `data/mcf_jobs.db`)

### Historical Scraper (Full Archive)

Enumerate and fetch all historical MCF jobs by job ID. Uses UUID = MD5(jobPostId) to access closed/historical jobs.

```bash
# Scrape a specific year
python -m src.cli scrape-historical --year 2023

# Scrape all years (2019-2026)
python -m src.cli scrape-historical --all

# Scrape a specific range
python -m src.cli scrape-historical --start MCF-2023-0500000 --end MCF-2023-0600000

# Check progress
python -m src.cli historical-status

# Resume interrupted scrape
python -m src.cli scrape-historical --year 2023
```

**Historical Scrape Options:**
- `--year, -y` - Specific year to scrape (2019-2026)
- `--all` - Scrape all years
- `--start` - Starting jobPostId (e.g., MCF-2023-0500000)
- `--end` - Ending jobPostId
- `--resume/--no-resume` - Resume from checkpoint (default: resume)
- `--rate-limit, -r` - Requests per second (default: 2.0)
- `--not-found-threshold` - Stop after N consecutive not-found (default: 1000)
- `--dry-run` - Preview without fetching
- `--verbose, -v` - Debug logging

**Estimated Job Counts:**
| Year | Est. Jobs |
|------|-----------|
| 2019 | ~50K |
| 2020 | ~350K |
| 2021 | ~700K |
| 2022 | ~1M |
| 2023 | ~1M |
| 2024 | ~1.4M |
| 2025 | ~1.5M |
| 2026 | ~250K+ |
| **Total** | **~6.2M** |

**Runtime:** At 2 req/sec: ~36 days. At 5 req/sec: ~14 days.

### Daemon Mode (Long-Running Background Scrape)

Run the historical scraper as a background daemon that survives terminal closure and detects sleep/wake cycles:

```bash
# Start scraper as background daemon
python -m src.cli daemon start --year 2023
python -m src.cli daemon start --all

# Check daemon status
python -m src.cli daemon status

# Stop daemon
python -m src.cli daemon stop
```

**Daemon Features:**
- Runs in background via Unix fork (survives terminal close)
- Heartbeat-based wake detection (logs warning after laptop sleep)
- PID file at `data/.scraper.pid`
- Logs to `data/scraper_daemon.log`

### Gap Analysis and Retry

Ensure completeness by finding and retrying missed/failed job IDs:

```bash
# Show gaps in scraped data
python -m src.cli gaps --year 2023
python -m src.cli gaps --all

# Retry failed/missing IDs
python -m src.cli retry-gaps --year 2023
python -m src.cli retry-gaps --all

# View fetch attempt statistics
python -m src.cli attempt-stats
python -m src.cli attempt-stats --year 2023
```

**Gap Analysis Features:**
- Per-ID attempt tracking in `fetch_attempts` table
- Detects sequence gaps and failed fetches
- Retries only missing IDs (not full rescan)

### Legacy Scrapers (Slow, kept for reference)

```bash
python src/legacy/mycareersfuture.py "Machine Learning"
python src/linkedin.py "Data Scientist" "Singapore"
```

## Architecture

### New API-Based Scraper (`src/mcf/`)

The MCF API scraper uses the public MyCareersFuture REST API instead of browser automation:

```
src/
├── mcf/                      # API-based scraper package
│   ├── __init__.py           # Package exports
│   ├── api_client.py         # Async HTTP client with retry logic
│   ├── database.py           # SQLite storage with history tracking
│   ├── historical_scraper.py # Historical job enumeration by ID
│   ├── models.py             # Pydantic data models
│   ├── scraper.py            # Search-based scraping with resume
│   ├── storage.py            # JobStorage + SQLiteStorage classes
│   ├── batch_logger.py       # Per-ID attempt logging (batched)
│   ├── adaptive_rate.py      # Dynamic rate limiting
│   └── daemon.py             # Background process management
├── cli.py                    # Typer CLI interface
└── legacy/                   # Old Selenium scrapers
```

**Key Features:**
- **Fast**: ~1-2 minutes for 2,000 jobs (vs ~4 hours with Selenium)
- **Reliable**: Automatic retry with exponential backoff
- **Resumable**: Session-based resume after interruption
- **Deduplicated**: Same job won't appear twice across queries
- **Type-safe**: Pydantic models validate all data
- **SQLite Storage**: Persistent database with history tracking
- **History Tracking**: Detects and records changes when jobs are re-scraped
- **Per-ID Tracking**: Every fetch attempt logged for gap detection
- **Adaptive Rate Limiting**: Backs off on 429s, recovers slowly
- **Daemon Mode**: Background operation with wake detection

**API Endpoint:** `https://api.mycareersfuture.gov.sg/v2/jobs/`

### SQLite Database Schema

Data is stored in `data/mcf_jobs.db` with six tables:

#### `jobs` - Current state of each job listing

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| uuid | TEXT | Unique job identifier (indexed) |
| title | TEXT | Job title |
| company_name | TEXT | Company name (indexed) |
| company_uen | TEXT | Company unique entity number |
| description | TEXT | Job description (HTML stripped) |
| salary_min | INTEGER | Minimum salary (indexed) |
| salary_max | INTEGER | Maximum salary (indexed) |
| salary_type | TEXT | Monthly/Yearly/Hourly |
| employment_type | TEXT | Full Time/Part Time/Contract (indexed) |
| seniority | TEXT | Position level |
| min_experience_years | INTEGER | Minimum years experience |
| skills | TEXT | Comma-separated skills |
| categories | TEXT | Job categories |
| location | TEXT | Formatted address |
| district | TEXT | District name |
| region | TEXT | Region (Central/North/etc) |
| posted_date | DATE | Posting date (indexed) |
| expiry_date | DATE | Expiry date |
| applications_count | INTEGER | Number of applicants |
| job_url | TEXT | URL to job page |
| first_seen_at | TIMESTAMP | When job was first scraped |
| last_updated_at | TIMESTAMP | Last time job was updated |

#### `job_history` - Previous versions when jobs are updated

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| job_uuid | TEXT | References jobs.uuid (indexed) |
| title | TEXT | Previous title |
| company_name | TEXT | Previous company name |
| salary_min | INTEGER | Previous minimum salary |
| salary_max | INTEGER | Previous maximum salary |
| applications_count | INTEGER | Previous application count |
| description | TEXT | Previous description |
| recorded_at | TIMESTAMP | When this snapshot was taken |

#### `scrape_sessions` - Tracks search-based scraping progress

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| search_query | TEXT | Search query (indexed) |
| total_jobs | INTEGER | Total jobs available |
| fetched_count | INTEGER | Jobs fetched so far |
| current_offset | INTEGER | Current pagination offset |
| status | TEXT | in_progress/completed/interrupted (indexed) |
| started_at | TIMESTAMP | Session start time |
| updated_at | TIMESTAMP | Last progress update |
| completed_at | TIMESTAMP | When session completed |

#### `historical_scrape_progress` - Tracks historical job enumeration

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| year | INTEGER | Year being scraped (indexed) |
| start_seq | INTEGER | Starting sequence number |
| current_seq | INTEGER | Current sequence position |
| end_seq | INTEGER | Target ending sequence |
| jobs_found | INTEGER | Jobs successfully fetched |
| jobs_not_found | INTEGER | IDs that returned 404 |
| consecutive_not_found | INTEGER | Track end-of-year detection |
| status | TEXT | in_progress/completed/interrupted (indexed) |
| started_at | TIMESTAMP | Session start time |
| updated_at | TIMESTAMP | Last checkpoint |
| completed_at | TIMESTAMP | When session completed |

#### `fetch_attempts` - Per-ID attempt tracking for gap detection

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| year | INTEGER | Year of job ID (indexed) |
| sequence | INTEGER | Sequence number (indexed) |
| result | TEXT | found/not_found/error/skipped (indexed) |
| error_message | TEXT | Error details for 'error' results |
| attempted_at | TIMESTAMP | When attempt was made |

**Unique constraint:** `(year, sequence)` - One record per ID (upsert on retry)

#### `daemon_state` - Background process tracking

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Always 1 (singleton row) |
| pid | INTEGER | Process ID of running daemon |
| status | TEXT | running/stopped/sleeping |
| last_heartbeat | TIMESTAMP | Last heartbeat timestamp |
| started_at | TIMESTAMP | When daemon started |
| current_year | INTEGER | Year being scraped |
| current_seq | INTEGER | Current sequence position |

### Key Classes

**`MCFDatabase`** - Low-level SQLite operations
- `upsert_job(job)` - Insert or update, returns `(is_new, was_updated)`
- `search_jobs(**filters)` - Query with filters
- `get_job_history(uuid)` - Get change history
- `get_stats()` - Database statistics
- `export_to_csv(path, **filters)` - Export to CSV

**`SQLiteStorage`** - High-level storage with session tracking
- Same interface as `JobStorage` for compatibility
- Uses `MCFDatabase` for persistence
- Maintains in-memory cache for fast duplicate checks

**`MCFScraper`** - Orchestrates search-based scraping
- `use_sqlite=True` (default) for SQLite storage
- `use_sqlite=False` for legacy JSON checkpoints

**`HistoricalScraper`** - Enumerates historical jobs by ID
- `job_id_to_uuid(job_id)` - Convert MCF-YYYY-NNNNNNN to UUID (MD5)
- `scrape_year(year)` - Scrape all jobs for a year with checkpointing
- `scrape_range(start_id, end_id)` - Scrape specific ID range
- `scrape_all_years()` - Scrape 2019-2026
- `find_year_bounds(year)` - Binary search to find valid ID range
- `retry_gaps(year)` - Retry missing/failed sequences
- Uses `BatchLogger` for per-ID tracking and `AdaptiveRateLimiter` for dynamic rate control

**`BatchLogger`** - Efficient per-ID attempt tracking
- Buffers attempts in memory, flushes to SQLite every 50 entries
- Uses `atexit` to flush on exit
- Trade-off: Max 50 IDs lost on crash, but 50x less I/O

**`AdaptiveRateLimiter`** - Dynamic rate limiting
- Starts at 2 req/sec (configurable)
- On 429: Cut rate by 50% (min 0.5 req/sec)
- After 50 consecutive successes: Increase 10% (max 5 req/sec)
- Observable state via `get_state()`

**`ScraperDaemon`** - Background process manager
- Unix fork-based daemonization
- Heartbeat every 10s for wake detection
- 5-minute gap = sleep detected (logs warning)
- PID file management for start/stop/status

## Environment

- Python 3.10+ with Poetry for dependency management
- Virtual environment: `poetry shell` or `poetry run`

## Key Dependencies

**API Scraper:**
- `httpx` - Async HTTP client
- `pydantic` - Data validation
- `tenacity` - Retry logic
- `typer` - CLI framework
- `rich` - Terminal UI and progress bars
- `pandas` - Data manipulation and CSV export
- `sqlite3` - Database (Python stdlib)

**Legacy Scrapers:**
- `selenium` - Browser automation
- `aiohttp` / `arsenic` - Async variants

## Development

```bash
# Install all dependencies (including dev)
poetry install

# Run tests
poetry run pytest

# Run with verbose logging
python -m src.cli scrape "test" -v

# Test database operations
python -c "from src.mcf import MCFDatabase; db = MCFDatabase(); print(db.get_stats())"
```

## Configuration

### Adaptive Rate Limiter Defaults

| Parameter | Value | Description |
|-----------|-------|-------------|
| initial_rps | 2.0 | Starting requests per second |
| min_rps | 0.5 | Floor on rate (never slower) |
| max_rps | 5.0 | Ceiling on rate (never faster) |
| backoff_factor | 0.5 | Multiply RPS by this on 429 |
| recovery_factor | 1.1 | Multiply RPS by this on recovery |
| recovery_threshold | 50 | Successes before rate increase |

### Daemon Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| heartbeat_interval | 10s | Time between heartbeats |
| wake_threshold | 300s | Gap indicating sleep/wake (5 min) |
| batch_size | 50 | Attempts buffered before flush |
| pidfile | `data/.scraper.pid` | PID file location |
| logfile | `data/scraper_daemon.log` | Daemon log file |

## Known Issues

- LinkedIn scraper requires manual login handling
- Legacy scrapers have hardcoded ChromeDriver paths
- Colabctl module is broken due to Google Colab UI changes
- Daemon mode only works on Unix-like systems (uses `os.fork`)
