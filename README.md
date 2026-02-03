# MyCareersFuture Job Scraper

A fast, reliable job market data collection system for MyCareersFuture.sg. Collects job listings for tech roles (Data Science, Machine Learning, Data Engineering) with automatic deduplication and history tracking.

## Features

- **Fast API-based scraping** - ~1-2 minutes for 2,000 jobs
- **Historical archive access** - Enumerate all ~6.2M jobs from 2019-present
- **SQLite storage** - Persistent database with automatic deduplication
- **History tracking** - Records changes when jobs are re-scraped (salary updates, application counts)
- **Resumable** - Automatically resumes interrupted scrapes
- **Daemon mode** - Long-running background scraping with sleep/wake detection
- **Adaptive rate limiting** - Backs off on 429s, recovers slowly after success
- **Per-ID tracking** - Every fetch attempt logged for gap detection and retry
- **Query interface** - Search, filter, and export data via CLI
- **CSV/JSON export** - Export filtered subsets for analysis

## Quick Start

```bash
# Install dependencies
poetry install

# Preview search results
python -m src.cli preview "data scientist"

# Scrape jobs (stores in SQLite + exports CSV)
python -m src.cli scrape "data scientist"

# Query the database
python -m src.cli stats
python -m src.cli list --limit 20
python -m src.cli search "machine learning"
```

## Installation

Requires Python 3.10+

```bash
# Clone the repository
git clone <repo-url>
cd MyCareersFuture

# Install with Poetry
poetry install

# Or with pip
pip install -e .
```

## CLI Commands

### Scraping

```bash
# Scrape jobs for a search query
python -m src.cli scrape "data scientist"
python -m src.cli scrape "machine learning" --max-jobs 500

# Scrape multiple queries (deduplicated)
python -m src.cli scrape-multi "data scientist" "ML engineer" "data engineer"

# Options
#   --max-jobs, -n     Limit number of jobs
#   --output, -o       Output directory (default: data/)
#   --format, -f       Output format: csv or json
#   --no-resume        Don't resume from checkpoint
#   --rate-limit, -r   Requests per second (default: 2.0)
#   --verbose, -v      Enable debug logging
```

### Historical Scraping

Enumerate and fetch all historical jobs from the MCF archive (2019-present) by job ID:

```bash
# Scrape a specific year
python -m src.cli scrape-historical --year 2023

# Scrape all years (2019-2026, ~6.2M jobs total)
python -m src.cli scrape-historical --all

# Scrape a specific range
python -m src.cli scrape-historical --start MCF-2023-0500000 --end MCF-2023-0600000

# Check progress
python -m src.cli historical-status

# Resume interrupted scrape (automatic)
python -m src.cli scrape-historical --year 2023

# Options
#   --year, -y              Specific year to scrape
#   --all                   Scrape all years
#   --start/--end           Specific job ID range
#   --resume/--no-resume    Resume from checkpoint (default: resume)
#   --rate-limit, -r        Requests per second (default: 2.0)
#   --not-found-threshold   Stop after N consecutive missing IDs (default: 1000)
#   --dry-run               Preview without fetching
```

**Note:** Full historical scrape takes 14-36 days depending on rate limit. Progress is checkpointed every 100 jobs for safe interruption/resume.

### Daemon Mode (Long-Running Background Scrape)

Run the historical scraper as a background daemon that survives terminal closure:

```bash
# Start scraping 2023 in background
python -m src.cli daemon start --year 2023

# Start scraping all years in background
python -m src.cli daemon start --all

# Check daemon status
python -m src.cli daemon status

# Stop the daemon
python -m src.cli daemon stop
```

**Features:**
- Survives terminal closure (Unix fork-based)
- Detects laptop sleep/wake cycles (logs warning, continues from checkpoint)
- PID file at `data/.scraper.pid`
- Logs to `data/scraper_daemon.log`
- Adaptive rate limiting adjusts automatically to API responses

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

Every job ID fetch is tracked in the `fetch_attempts` table with result status (found/not_found/error/skipped), enabling detection of gaps and targeted retries.

### Querying

```bash
# List jobs with filters
python -m src.cli list --limit 20
python -m src.cli list --company "Google" --salary-min 8000
python -m src.cli list --employment-type "Permanent"

# Search by keyword
python -m src.cli search "Python"
python -m src.cli search "machine learning" --limit 50

# Database statistics
python -m src.cli stats

# Export filtered data
python -m src.cli export jobs.csv
python -m src.cli export high_salary.csv --salary-min 10000

# View job change history
python -m src.cli history <job-uuid>

# Check scrape sessions
python -m src.cli db-status
```

## Data Storage

### SQLite Database

Jobs are stored in `data/mcf_jobs.db` with the following tables:

**`jobs`** - Current state of each job
- Job details: title, company, salary, skills, location
- Timestamps: `first_seen_at`, `last_updated_at`

**`job_history`** - Previous versions when jobs change
- Tracks salary changes, application count increases
- Useful for analyzing job market trends

**`scrape_sessions`** - Progress tracking for search-based scrapes

**`historical_scrape_progress`** - Progress tracking for historical enumeration
- Tracks year, current sequence, jobs found/not found

**`fetch_attempts`** - Per-ID attempt logging for completeness
- Every fetch attempt with result: found, not_found, error, skipped
- Enables gap detection and targeted retries

**`daemon_state`** - Background daemon tracking
- PID, status, last heartbeat, current position

### Output Schema

| Field | Type | Description |
|-------|------|-------------|
| uuid | str | Unique job identifier |
| title | str | Job title |
| company_name | str | Company name |
| company_uen | str | Company UEN |
| description | str | Job description |
| salary_min | int | Minimum salary |
| salary_max | int | Maximum salary |
| salary_type | str | Monthly/Yearly/Hourly |
| employment_type | str | Full Time/Part Time/Contract |
| seniority | str | Position level |
| min_experience_years | int | Required experience |
| skills | str | Comma-separated skills |
| categories | str | Job categories |
| location | str | Address |
| district | str | District |
| region | str | Region |
| posted_date | date | When posted |
| expiry_date | date | Expiry date |
| applications_count | int | Number of applicants |
| job_url | str | Link to job page |

## Architecture

```
src/
├── mcf/                      # Main scraper package
│   ├── api_client.py         # Async HTTP client with retry
│   ├── database.py           # SQLite operations + new tables
│   ├── historical_scraper.py # Historical job enumeration
│   ├── models.py             # Pydantic models
│   ├── scraper.py            # Search-based scraping
│   ├── storage.py            # Storage classes
│   ├── batch_logger.py       # Per-ID attempt logging (batched)
│   ├── adaptive_rate.py      # Dynamic rate limiting
│   └── daemon.py             # Background process manager
├── cli.py                    # CLI interface
└── legacy/                   # Old Selenium scrapers
```

### Key Classes

```python
from src.mcf import MCFScraper, MCFDatabase, HistoricalScraper

# Scrape jobs by search query
scraper = MCFScraper()
jobs = await scraper.scrape("data scientist", max_jobs=100)
scraper.save("data_scientist")

# Query database directly
db = MCFDatabase()
jobs = db.search_jobs(company_name="Google", salary_min=8000)
stats = db.get_stats()
history = db.get_job_history("job-uuid")

# Historical scraping with adaptive rate limiting
async with HistoricalScraper() as scraper:
    await scraper.scrape_year(2023)
    await scraper.retry_gaps(2023)  # Retry any missed IDs
```

### Robust Pipeline Components

**`AdaptiveRateLimiter`** - Dynamic rate control
- Starts at 2 req/sec, backs off 50% on 429
- Recovers 10% after 50 consecutive successes
- Min 0.5 req/sec, max 5 req/sec

**`BatchLogger`** - Efficient attempt tracking
- Buffers 50 attempts in memory before flushing to SQLite
- Uses atexit to flush on exit, minimizing data loss

**`ScraperDaemon`** - Background process manager
- Unix double-fork for proper daemonization
- Heartbeat every 10s, detects 5+ minute gaps as sleep/wake

## Development

```bash
# Install dev dependencies
poetry install

# Run tests
poetry run pytest

# Run with verbose logging
python -m src.cli scrape "test" -v
```

## License

MIT
