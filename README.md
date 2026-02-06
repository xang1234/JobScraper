# MyCareersFuture Job Market Intelligence Platform

A job market data collection and semantic search platform for MyCareersFuture.sg. Scrapes job listings, generates vector embeddings, and serves hybrid BM25 + FAISS search through a REST API with a React frontend.

## Features

**Data Collection**
- Fast API-based scraping — ~1-2 minutes for 2,000 jobs
- Historical archive access — enumerate all ~6.2M jobs from 2019-present
- SQLite storage with automatic deduplication and history tracking
- Resumable scrapes, daemon mode, adaptive rate limiting, per-ID tracking

**Semantic Search**
- Hybrid ranking combining FAISS vector similarity + BM25 keyword matching
- Query expansion with skill-cluster synonyms
- Freshness boosting for recent postings
- Skill similarity search, related skills discovery, company similarity
- Degraded mode (keyword-only) when indexes are unavailable

**REST API**
- 12 endpoints covering search, recommendations, skills, analytics
- Rate limiting, CORS, request logging middleware
- Interactive Swagger docs at `/docs`

**Deployment**
- Docker Compose with backend + frontend/nginx services
- Production overrides with resource limits and bind mounts
- Data bootstrap script for seeding containers from local data

## Quick Start

```bash
# Install dependencies
poetry install

# Scrape some jobs
poetry run python -m src.cli scrape "data scientist"

# Generate embeddings and build search indexes
poetry run python -m src.cli embed-generate

# Semantic search from CLI
poetry run python -m src.cli search-semantic "machine learning engineer"

# Start API server (Swagger UI at http://localhost:8000/docs)
poetry run python -m src.cli api-serve --reload

# Or run everything with Docker
docker compose up
```

> **Tip:** Add an alias for convenience: `alias mcf="poetry run python -m src.cli"`

## Installation

Requires Python 3.10+

```bash
# Clone the repository
git clone https://github.com/xang1234/JobScraper.git
cd JobScraper

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

### Monitoring Progress

Several commands help you monitor scraping progress:

**Historical scrape status** - Shows progress per year with jobs found:
```bash
python -m src.cli historical-status
```

```
Historical Scrape Status

Active Sessions:

┏━━━━┳━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃    ┃      ┃               ┃       ┃           ┃  Consecutive ┃               ┃
┃ ID ┃ Year ┃      Progress ┃ Found ┃ Not Found ┃           NF ┃ Started       ┃
┡━━━━╇━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ 6  │ 2022 │ 245/1,000,000 │   243 │         0 │            0 │ 2026-02-03    │
│    │      │        (0.0%) │       │           │              │ 00:39         │
│ 7  │ 2019 │    885/50,000 │   862 │        21 │            0 │ 2026-02-03    │
│    │      │        (1.8%) │       │           │              │ 02:50         │
└────┴──────┴───────────────┴───────┴───────────┴──────────────┴───────────────┘

Jobs in Database by Year:

┏━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Year ┃  Jobs ┃ Est. Total ┃ Coverage ┃
┡━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ 2024 │     9 │ ~1,450,000 │     0.0% │
│ 2023 │ 4,246 │ ~1,000,000 │     0.4% │
│ 2022 │   200 │ ~1,000,000 │     0.0% │
│ 2019 │   862 │    ~50,000 │     1.7% │
└──────┴───────┴────────────┴──────────┘
```

**Fetch attempt statistics** - Shows found/not_found/error counts:
```bash
python -m src.cli attempt-stats
```

```
Fetch Attempt Statistics

┏━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Year ┃ Total ┃ Found ┃ Not Found ┃ Skipped ┃ Errors ┃
┡━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│ 2019 │   883 │   862 │        21 │       0 │      0 │
└──────┴───────┴───────┴───────────┴─────────┴────────┘

Grand total: 883 attempts, 862 jobs found
```

**Database statistics** - Shows totals, salary ranges, top companies:
```bash
python -m src.cli stats
```

```
Database Statistics

Total jobs: 6,361
Jobs with history: 0
History records: 0
Added today: 6,361
Updated today: 0

Salary range: $1 - $500,000
Average range: $5,794 - $9,328

By Employment Type:
  Permanent: 2,299
  Full Time: 2,101
  Contract: 698

Top Companies:
  DBS BANK LTD.: 264 jobs
  THE SUPREME HR ADVISORY PTE. LTD.: 251 jobs
  TIKTOK PTE. LTD.: 154 jobs
```

**Live daemon logs:**
```bash
tail -f data/scraper_daemon.log
```

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

### Semantic Search

Hybrid search combining vector similarity with keyword matching. Requires embeddings to be generated first.

```bash
# Basic semantic search
python -m src.cli search-semantic "machine learning engineer"

# With filters
python -m src.cli search-semantic "python developer" --salary-min 8000
python -m src.cli search-semantic "data scientist" --company Google

# Tune search behavior
python -m src.cli search-semantic "AI engineer" --alpha 0.5   # More keyword weight
python -m src.cli search-semantic "ML" --no-expand             # Disable query expansion

# JSON output for scripting
python -m src.cli search-semantic "data engineer" --json

# Options
#   --limit, -n           Number of results (default: 10)
#   --salary-min          Minimum salary filter
#   --salary-max          Maximum salary filter
#   --company, -c         Company name filter
#   --employment-type, -e Employment type filter
#   --region, -r          Region filter
#   --alpha               Semantic vs keyword weight (0=keyword, 1=semantic, default: 0.7)
#   --no-expand           Disable query expansion with skill synonyms
#   --json                Output as JSON
```

### Embeddings

Generate vector embeddings and build FAISS indexes for semantic search:

```bash
# Generate embeddings and build FAISS indexes (full run)
python -m src.cli embed-generate

# Generate embeddings only (skip index building)
python -m src.cli embed-generate --no-build-index

# Regenerate all embeddings (ignore existing)
python -m src.cli embed-generate --no-skip-existing

# Sync new jobs and update indexes incrementally
python -m src.cli embed-sync

# Check embedding and index status
python -m src.cli embed-status

# Upgrade to a new embedding model
python -m src.cli embed-upgrade all-mpnet-base-v2 --yes

# Options (embed-generate / embed-sync)
#   --batch-size, -b                 Jobs per batch (default: 32)
#   --skip-existing/--no-skip-existing  Skip jobs with embeddings (default: skip)
#   --build-index/--no-build-index   Build FAISS indexes after generation (default: build)
#   --update-index/--no-update-index Update FAISS indexes on sync (default: update)
#   --index-dir                      FAISS index directory (default: data/embeddings)
#   --db                             Database path (default: data/mcf_jobs.db)
```

### REST API

Start the semantic search API server:

```bash
# Start on localhost:8000 (Swagger UI at /docs)
python -m src.cli api-serve

# Development mode with auto-reload
python -m src.cli api-serve --reload

# Production mode with multiple workers
python -m src.cli api-serve --workers 4

# Custom host/port
python -m src.cli api-serve --host 0.0.0.0 --port 9000

# Options
#   --host, -H       Host to bind to (default: 127.0.0.1)
#   --port, -p       Port to bind to (default: 8000)
#   --reload         Auto-reload for development
#   --workers, -w    Number of worker processes (default: 1)
#   --cors           Comma-separated CORS origins
#   --rate-limit     Max requests per minute per IP (default: 100, 0 to disable)
#   --db             Database path
#   --index-dir      FAISS index directory
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Hybrid semantic + keyword job search |
| `/api/similar` | POST | Find jobs similar to a given UUID |
| `/api/similar/batch` | POST | Batch similar jobs lookup (max 50 UUIDs) |
| `/api/search/skills` | POST | Search jobs by skill similarity |
| `/api/skills/cloud` | GET | Skill frequency data for visualization |
| `/api/skills/related/{skill}` | GET | Related skills with similarity scores |
| `/api/companies/similar` | POST | Find companies with similar job profiles |
| `/api/stats` | GET | System statistics (index size, coverage) |
| `/api/analytics/popular` | GET | Popular search queries |
| `/api/analytics/performance` | GET | Search latency percentiles (p50/p90/p95/p99) |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

#### Example: Search Request

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning engineer",
    "limit": 5,
    "salary_min": 10000,
    "alpha": 0.7
  }'
```

Response:

```json
{
  "results": [
    {
      "uuid": "abc-123",
      "title": "ML Engineer",
      "company_name": "Google",
      "description": "We are looking for...",
      "salary_min": 12000,
      "salary_max": 18000,
      "employment_type": "Full Time",
      "skills": "Python, TensorFlow, PyTorch",
      "similarity_score": 0.923
    }
  ],
  "total_candidates": 1234,
  "search_time_ms": 47.2,
  "degraded": false,
  "cache_hit": false
}
```

### Docker Deployment

Run the full stack (backend API + frontend/nginx) with Docker Compose:

```bash
# Start all services (frontend at :3000, API at :8000)
docker compose up

# Build and start in background
docker compose up -d --build

# Bootstrap local data into the container
./docker/bootstrap-data.sh

# Production mode with resource limits and bind mounts
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
./docker/bootstrap-data.sh --prod
```

**Services:**

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| `backend` | `mcf-backend` | 8000 | FastAPI + uvicorn with FAISS indexes |
| `frontend` | `mcf-frontend` | 3000 | React app served by nginx |

The frontend container waits for the backend health check to pass before starting. Data is stored in Docker volumes (`mcf-data`, `hf-cache`) or bind-mounted to `/opt/mcf/` in production mode.

### Benchmarks

Run performance benchmarks on the search system:

```bash
python -m src.cli benchmark --queries 50 --warmup 5
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

**`embeddings`** - Vector embeddings for jobs and skills
- Used by the semantic search engine and FAISS index builder

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
├── api/                          # FastAPI REST API
│   ├── app.py                    # Route definitions + app factory
│   ├── models.py                 # Request/response Pydantic models
│   └── middleware.py             # Rate limiting, request logging
├── mcf/                          # Core scraper + search package
│   ├── api_client.py             # Async HTTP client with retry
│   ├── database.py               # SQLite operations
│   ├── historical_scraper.py     # Historical job enumeration
│   ├── models.py                 # Pydantic scraper models
│   ├── scraper.py                # Search-based scraping
│   ├── storage.py                # Storage classes
│   ├── batch_logger.py           # Per-ID attempt logging (batched)
│   ├── adaptive_rate.py          # Dynamic rate limiting
│   ├── daemon.py                 # Background process manager
│   └── embeddings/               # Semantic search engine
│       ├── search_engine.py      # Hybrid BM25 + FAISS orchestrator
│       ├── index_manager.py      # FAISS index build/load/save
│       ├── generator.py          # Embedding generation + clustering
│       ├── query_expander.py     # Skill-cluster query expansion
│       └── models.py             # Search dataclasses
├── frontend/                     # React + Vite frontend
│   ├── src/                      # React components and pages
│   ├── vite.config.ts            # Vite build configuration
│   └── package.json              # Node.js dependencies
├── cli.py                        # Typer CLI interface
└── legacy/                       # Old Selenium scrapers
docker/
├── backend.Dockerfile            # Multi-stage Python build
├── frontend.Dockerfile           # Multi-stage Node + nginx build
├── nginx.conf                    # Reverse proxy configuration
└── bootstrap-data.sh             # Data seeding script
docker-compose.yml                # Local development stack
docker-compose.prod.yml           # Production overrides
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

```python
from src.mcf.embeddings import SemanticSearchEngine, SearchRequest

# Semantic search
engine = SemanticSearchEngine("data/mcf_jobs.db", Path("data/embeddings"))
engine.load()

response = engine.search(SearchRequest(
    query="machine learning engineer",
    salary_min=10000,
    alpha=0.7,
))

for job in response.results:
    print(f"{job.title} @ {job.company_name} — {job.similarity_score:.3f}")
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

**`SemanticSearchEngine`** - Hybrid search orchestrator
- SQL pre-filtering → FAISS vector retrieval → BM25 re-ranking → freshness boost
- LRU cache for repeated queries
- Graceful degradation to keyword-only when indexes are missing

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
