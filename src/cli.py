#!/usr/bin/env python3
"""
CLI interface for the MyCareersFuture job scraper.

Usage:
    python -m src.cli scrape "data scientist"
    python -m src.cli scrape "machine learning" --max-jobs 500
    python -m src.cli status
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.mcf import (
    MCFClient,
    MCFScraper,
    JobStorage,
    SQLiteStorage,
    MCFDatabase,
    MCFMigrator,
    HistoricalScraper,
    YEAR_ESTIMATES,
    ScraperDaemon,
    DaemonAlreadyRunning,
    DaemonNotRunning,
    EmbeddingGenerator,
    EmbeddingStats,
)
from src.mcf.embeddings import (
    FAISSIndexManager,
    IndexCompatibilityError,
    SemanticSearchEngine,
    SearchRequest,
    SearchResponse,
)

app = typer.Typer(
    name="mcf",
    help="MyCareersFuture job scraper - fast API-based job data collection",
    add_completion=False,
)
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@app.command()
def scrape(
    query: str = typer.Argument(..., help="Search query (e.g., 'data scientist')"),
    max_jobs: Optional[int] = typer.Option(
        None, "--max-jobs", "-n", help="Maximum number of jobs to scrape"
    ),
    output_dir: str = typer.Option(
        "data", "--output", "-o", help="Output directory for files"
    ),
    format: str = typer.Option(
        "csv", "--format", "-f", help="Output format: csv or json"
    ),
    no_resume: bool = typer.Option(
        False, "--no-resume", help="Don't resume from previous checkpoint"
    ),
    rate_limit: float = typer.Option(
        2.0, "--rate-limit", "-r", help="Requests per second"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Scrape job listings from MyCareersFuture.

    Examples:
        mcf scrape "data scientist"
        mcf scrape "machine learning" --max-jobs 500
        mcf scrape "data engineer" -o ./jobs -f json
    """
    setup_logging(verbose)

    console.print(f"\n[bold blue]MCF Job Scraper[/bold blue]")
    console.print(f"Search query: [green]{query}[/green]")

    if max_jobs:
        console.print(f"Max jobs: {max_jobs}")

    console.print()

    async def run():
        scraper = MCFScraper(
            output_dir=output_dir,
            requests_per_second=rate_limit,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            await scraper.scrape(
                query,
                max_jobs=max_jobs,
                resume=not no_resume,
                progress=progress,
            )

        # Save results
        if scraper.job_count > 0:
            filepath = scraper.save(query, format=format)
            console.print(f"\n[green]Success![/green] Saved {scraper.job_count} jobs to {filepath}")

            # Show sample
            df = scraper.get_dataframe()
            if len(df) > 0:
                console.print("\n[bold]Sample jobs:[/bold]")
                table = Table(show_header=True)
                table.add_column("Title", style="cyan", max_width=40)
                table.add_column("Company", max_width=30)
                table.add_column("Salary", justify="right")

                for _, row in df.head(5).iterrows():
                    salary = ""
                    if row.get("salary_min") and row.get("salary_max"):
                        salary = f"${row['salary_min']:,} - ${row['salary_max']:,}"
                    table.add_row(
                        str(row.get("title", ""))[:40],
                        str(row.get("company_name", ""))[:30],
                        salary,
                    )

                console.print(table)
        else:
            console.print("[yellow]No jobs found[/yellow]")

    asyncio.run(run())


@app.command()
def scrape_multi(
    queries: list[str] = typer.Argument(..., help="Search queries to scrape"),
    max_jobs: Optional[int] = typer.Option(
        None, "--max-jobs", "-n", help="Maximum jobs per query"
    ),
    output_dir: str = typer.Option(
        "data", "--output", "-o", help="Output directory"
    ),
    output_name: str = typer.Option(
        "jobs", "--name", help="Base name for output file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """
    Scrape multiple search queries with deduplication.

    Example:
        mcf scrape-multi "data scientist" "machine learning" "data engineer"
    """
    setup_logging(verbose)

    console.print(f"\n[bold blue]MCF Multi-Query Scraper[/bold blue]")
    console.print(f"Queries: {', '.join(queries)}")
    console.print()

    async def run():
        scraper = MCFScraper(output_dir=output_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            await scraper.scrape_multiple(
                queries,
                max_jobs_per_query=max_jobs,
                progress=progress,
            )

        if scraper.job_count > 0:
            filepath = scraper.save(output_name)
            console.print(f"\n[green]Success![/green] Saved {scraper.job_count} unique jobs to {filepath}")

    asyncio.run(run())


@app.command()
def status() -> None:
    """
    Show status of pending checkpoints (incomplete scrapes).
    """
    checkpoint_dir = Path(".mcf_checkpoints")

    if not checkpoint_dir.exists():
        console.print("No checkpoints found")
        return

    checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))

    if not checkpoints:
        console.print("No incomplete scrapes found")
        return

    console.print("\n[bold]Incomplete Scrapes:[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Query")
    table.add_column("Progress", justify="right")
    table.add_column("Last Updated")

    import json
    from datetime import datetime

    for cp_file in checkpoints:
        try:
            with open(cp_file) as f:
                data = json.load(f)

            query = data.get("search_query", "unknown")
            fetched = data.get("fetched_count", 0)
            total = data.get("total_jobs", 0)
            updated = data.get("updated_at", "")

            if updated:
                try:
                    dt = datetime.fromisoformat(updated)
                    updated = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass

            progress_pct = (fetched / total * 100) if total > 0 else 0
            progress_str = f"{fetched}/{total} ({progress_pct:.1f}%)"

            table.add_row(query, progress_str, updated)
        except Exception as e:
            console.print(f"Error reading {cp_file}: {e}")

    console.print(table)
    console.print("\nRun [bold]mcf scrape <query>[/bold] to resume")


@app.command()
def preview(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of jobs to show"),
) -> None:
    """
    Preview job search results without saving.
    """
    setup_logging(verbose=False)

    console.print(f"\n[bold blue]Preview: {query}[/bold blue]\n")

    async def run():
        async with MCFClient() as client:
            response = await client.search_jobs(query, limit=limit)

            console.print(f"Found {response.total} total jobs\n")

            if not response.results:
                console.print("[yellow]No results[/yellow]")
                return

            for job in response.results:
                salary = ""
                if job.salary_min and job.salary_max:
                    salary = f"${job.salary_min:,} - ${job.salary_max:,} {job.salary_type}"

                console.print(f"[bold cyan]{job.title}[/bold cyan]")
                console.print(f"  Company: {job.company_name}")
                if salary:
                    console.print(f"  Salary: {salary}")
                console.print(f"  Type: {job.employment_type} | Level: {job.seniority}")
                if job.skills_list:
                    console.print(f"  Skills: {job.skills_list[:80]}...")
                console.print(f"  URL: {job.job_url}")
                console.print()

    asyncio.run(run())


@app.command()
def clear_checkpoints() -> None:
    """
    Clear all saved checkpoints.
    """
    checkpoint_dir = Path(".mcf_checkpoints")

    if not checkpoint_dir.exists():
        console.print("No checkpoints to clear")
        return

    import shutil
    shutil.rmtree(checkpoint_dir)
    console.print("[green]All checkpoints cleared[/green]")


# Database query commands


@app.command(name="list")
def list_jobs(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of jobs to show"),
    company: Optional[str] = typer.Option(None, "--company", "-c", help="Filter by company name"),
    salary_min: Optional[int] = typer.Option(None, "--salary-min", help="Minimum salary"),
    salary_max: Optional[int] = typer.Option(None, "--salary-max", help="Maximum salary"),
    employment_type: Optional[str] = typer.Option(None, "--employment-type", "-e", help="Employment type"),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
) -> None:
    """
    List jobs from the database with optional filters.

    Examples:
        mcf list --limit 20
        mcf list --company Google --salary-min 8000
        mcf list --employment-type "Full Time"
    """
    db = MCFDatabase(db_path)

    jobs = db.search_jobs(
        company_name=company,
        salary_min=salary_min,
        salary_max=salary_max,
        employment_type=employment_type,
        limit=limit,
    )

    if not jobs:
        console.print("[yellow]No jobs found matching filters[/yellow]")
        return

    console.print(f"\n[bold blue]Jobs ({len(jobs)} shown)[/bold blue]\n")

    table = Table(show_header=True)
    table.add_column("Title", style="cyan", max_width=35)
    table.add_column("Company", max_width=25)
    table.add_column("Salary", justify="right")
    table.add_column("Type", max_width=12)
    table.add_column("Posted")

    for job in jobs:
        salary = ""
        if job.get("salary_min") and job.get("salary_max"):
            salary = f"${job['salary_min']:,} - ${job['salary_max']:,}"

        table.add_row(
            str(job.get("title", ""))[:35],
            str(job.get("company_name", ""))[:25],
            salary,
            str(job.get("employment_type", ""))[:12],
            str(job.get("posted_date", ""))[:10],
        )

    console.print(table)


@app.command()
def search(
    keyword: str = typer.Argument(..., help="Search keyword"),
    field: str = typer.Option("all", "--field", "-f", help="Field to search: all, title, skills"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
) -> None:
    """
    Search jobs by keyword.

    Examples:
        mcf search "machine learning"
        mcf search "Python" --field skills
        mcf search "Senior" --field title --limit 50
    """
    db = MCFDatabase(db_path)

    jobs = db.search_jobs(keyword=keyword, limit=limit)

    if not jobs:
        console.print(f"[yellow]No jobs found matching '{keyword}'[/yellow]")
        return

    console.print(f"\n[bold blue]Search results for '{keyword}' ({len(jobs)} found)[/bold blue]\n")

    table = Table(show_header=True)
    table.add_column("Title", style="cyan", max_width=40)
    table.add_column("Company", max_width=25)
    table.add_column("Skills", max_width=40)

    for job in jobs:
        skills = str(job.get("skills", ""))
        if len(skills) > 40:
            skills = skills[:37] + "..."

        table.add_row(
            str(job.get("title", ""))[:40],
            str(job.get("company_name", ""))[:25],
            skills,
        )

    console.print(table)


@app.command()
def stats(
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
) -> None:
    """
    Show database statistics.
    """
    db = MCFDatabase(db_path)
    stats_data = db.get_stats()

    console.print("\n[bold blue]Database Statistics[/bold blue]\n")

    # General stats
    console.print(f"[bold]Total jobs:[/bold] {stats_data['total_jobs']:,}")
    console.print(f"[bold]Jobs with history:[/bold] {stats_data['jobs_with_history']:,}")
    console.print(f"[bold]History records:[/bold] {stats_data['history_records']:,}")
    console.print(f"[bold]Added today:[/bold] {stats_data['jobs_added_today']:,}")
    console.print(f"[bold]Updated today:[/bold] {stats_data['jobs_updated_today']:,}")

    # Salary stats
    salary = stats_data.get("salary_stats", {})
    if salary.get("min"):
        console.print(f"\n[bold]Salary range:[/bold] ${salary['min']:,} - ${salary['max']:,}")
        console.print(f"[bold]Average range:[/bold] ${salary['avg_min']:,} - ${salary['avg_max']:,}")

    # Employment types
    if stats_data.get("by_employment_type"):
        console.print("\n[bold]By Employment Type:[/bold]")
        for emp_type, count in stats_data["by_employment_type"].items():
            console.print(f"  {emp_type}: {count:,}")

    # Top companies
    if stats_data.get("top_companies"):
        console.print("\n[bold]Top Companies:[/bold]")
        for company, count in list(stats_data["top_companies"].items())[:5]:
            console.print(f"  {company}: {count:,} jobs")


@app.command()
def export(
    output: Path = typer.Argument(..., help="Output CSV file path"),
    keyword: Optional[str] = typer.Option(None, "--keyword", "-k", help="Filter by keyword"),
    company: Optional[str] = typer.Option(None, "--company", "-c", help="Filter by company"),
    salary_min: Optional[int] = typer.Option(None, "--salary-min", help="Minimum salary"),
    salary_max: Optional[int] = typer.Option(None, "--salary-max", help="Maximum salary"),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
) -> None:
    """
    Export jobs from database to CSV.

    Examples:
        mcf export jobs.csv
        mcf export high_salary.csv --salary-min 10000
        mcf export google_jobs.csv --company Google
    """
    db = MCFDatabase(db_path)

    count = db.export_to_csv(
        output,
        keyword=keyword,
        company_name=company,
        salary_min=salary_min,
        salary_max=salary_max,
    )

    if count > 0:
        console.print(f"[green]Exported {count:,} jobs to {output}[/green]")
    else:
        console.print("[yellow]No jobs found matching filters[/yellow]")


@app.command()
def history(
    uuid: str = typer.Argument(..., help="Job UUID"),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
) -> None:
    """
    Show history of changes for a job.

    Example:
        mcf history abc123-def456
    """
    db = MCFDatabase(db_path)

    # Get current job
    job = db.get_job(uuid)
    if not job:
        console.print(f"[red]Job not found: {uuid}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Job: {job['title']}[/bold blue]")
    console.print(f"Company: {job['company_name']}")
    console.print(f"First seen: {job['first_seen_at']}")
    console.print(f"Last updated: {job['last_updated_at']}")

    # Get history
    history_records = db.get_job_history(uuid)

    if not history_records:
        console.print("\n[yellow]No history records (job hasn't been updated)[/yellow]")
        return

    console.print(f"\n[bold]History ({len(history_records)} updates):[/bold]\n")

    table = Table(show_header=True)
    table.add_column("Date", style="dim")
    table.add_column("Title")
    table.add_column("Company")
    table.add_column("Salary")
    table.add_column("Applications")

    for record in history_records:
        salary = ""
        if record.get("salary_min") and record.get("salary_max"):
            salary = f"${record['salary_min']:,} - ${record['salary_max']:,}"

        table.add_row(
            str(record.get("recorded_at", ""))[:19],
            str(record.get("title", "")),
            str(record.get("company_name", "")),
            salary,
            str(record.get("applications_count", "")),
        )

    console.print(table)


@app.command()
def db_status(
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
) -> None:
    """
    Show status of scrape sessions in database.
    """
    db = MCFDatabase(db_path)
    sessions = db.get_all_sessions()

    if not sessions:
        console.print("No scrape sessions found")
        return

    console.print("\n[bold blue]Scrape Sessions[/bold blue]\n")

    table = Table(show_header=True)
    table.add_column("ID")
    table.add_column("Query")
    table.add_column("Progress", justify="right")
    table.add_column("Status")
    table.add_column("Started")

    for session in sessions[:20]:  # Show last 20
        progress_str = f"{session['fetched_count']}/{session['total_jobs']}"
        if session['total_jobs'] > 0:
            pct = session['fetched_count'] / session['total_jobs'] * 100
            progress_str += f" ({pct:.0f}%)"

        status_style = {
            "completed": "green",
            "in_progress": "yellow",
            "interrupted": "red",
        }.get(session["status"], "")

        table.add_row(
            str(session["id"]),
            session["search_query"],
            progress_str,
            f"[{status_style}]{session['status']}[/{status_style}]",
            str(session["started_at"])[:16],
        )

    console.print(table)


@app.command()
def migrate(
    data_dir: str = typer.Option("data", "--data-dir", "-d", help="Data directory"),
    json_only: bool = typer.Option(False, "--json-only", help="Only import JSON files"),
    csv_only: bool = typer.Option(False, "--csv-only", help="Only import CSV files"),
    skip_link_only: bool = typer.Option(
        False, "--skip-link-only", help="Skip link-only CSVs (minimal data)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview without importing"
    ),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """
    Import legacy MCF data from JSON files and CSVs into SQLite.

    Processes data sources in order for best data quality:
    1. JSON files first (richest data, from data/scrape_jsons/)
    2. Full MCF CSVs second (16 columns with full job data)
    3. Link-only CSVs last (minimal data: Company, Title, Location, Link)

    Examples:
        mcf migrate                    # Import all legacy data
        mcf migrate --json-only        # Only JSON files
        mcf migrate --dry-run          # Preview without importing
        mcf migrate --skip-link-only   # Skip minimal data CSVs
    """
    setup_logging(verbose)

    if dry_run:
        console.print("[bold yellow]DRY RUN[/bold yellow] - No data will be imported\n")

    console.print("[bold blue]MCF Legacy Data Migration[/bold blue]")
    console.print(f"Data directory: [green]{data_dir}[/green]")
    console.print(f"Database: [green]{db_path}[/green]")

    if json_only:
        console.print("Mode: JSON files only")
    elif csv_only:
        console.print("Mode: CSV files only")
    else:
        console.print("Mode: All sources (JSON + CSV)")

    if skip_link_only:
        console.print("Skipping: Link-only CSVs")

    console.print()

    # Get initial stats
    db = MCFDatabase(db_path)
    initial_count = db.count_jobs()
    console.print(f"Jobs in database before migration: [cyan]{initial_count:,}[/cyan]")
    console.print()

    # Run migration with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Migrating legacy data...", total=None)

        migrator = MCFMigrator(db_path)
        stats = migrator.migrate_all(
            data_dir=data_dir,
            json_only=json_only,
            csv_only=csv_only,
            skip_link_only=skip_link_only,
            dry_run=dry_run,
        )

        progress.update(task, completed=True)

    # Display results
    console.print("\n[bold green]Migration Complete![/bold green]\n")

    # Stats table
    table = Table(title="Migration Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")

    table.add_row("JSON files processed", f"{stats.json_files_processed:,}")
    table.add_row("CSV rows processed", f"{stats.csv_rows_processed:,}")
    table.add_row("New jobs imported", f"[green]{stats.new_jobs:,}[/green]")
    table.add_row("Jobs updated", f"[yellow]{stats.updated_jobs:,}[/yellow]")
    table.add_row("Skipped (duplicates)", f"{stats.skipped_duplicates:,}")
    table.add_row("Link-only records", f"{stats.link_only_jobs:,}")
    table.add_row("Errors", f"[red]{len(stats.errors):,}[/red]" if stats.errors else "0")

    console.print(table)

    # Final count
    if not dry_run:
        final_count = db.count_jobs()
        console.print(f"\nJobs in database after migration: [cyan]{final_count:,}[/cyan]")
        console.print(f"Net change: [green]+{final_count - initial_count:,}[/green]")

    # Show sample errors if any
    if stats.errors and verbose:
        console.print(f"\n[bold red]Sample Errors ({len(stats.errors)} total):[/bold red]")
        for error in stats.errors[:5]:
            console.print(f"  - {error.source}")
            if error.row is not None:
                console.print(f"    Row {error.row}: {error.error}")
            else:
                console.print(f"    {error.error}")


# Historical scraping commands


@app.command(name="scrape-historical")
def scrape_historical(
    year: Optional[int] = typer.Option(
        None, "--year", "-y", help="Specific year to scrape (2019-2026)"
    ),
    all_years: bool = typer.Option(
        False, "--all", help="Scrape all years (2019-2026)"
    ),
    start: Optional[str] = typer.Option(
        None, "--start", help="Starting jobPostId (e.g., MCF-2023-0500000)"
    ),
    end: Optional[str] = typer.Option(
        None, "--end", help="Ending jobPostId (e.g., MCF-2023-0600000)"
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from previous checkpoint"
    ),
    rate_limit: float = typer.Option(
        2.0, "--rate-limit", "-r", help="Requests per second"
    ),
    not_found_threshold: int = typer.Option(
        1000, "--not-found-threshold", help="Stop after N consecutive not-found"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview without fetching"
    ),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """
    Scrape historical jobs by enumerating job IDs.

    This scrapes jobs from the MCF archive by generating all possible job IDs
    (MCF-YYYY-NNNNNNN format) and fetching each one. Jobs are stored in SQLite
    with automatic deduplication.

    Examples:
        mcf scrape-historical --year 2023
        mcf scrape-historical --all
        mcf scrape-historical --start MCF-2023-0500000 --end MCF-2023-0600000
        mcf scrape-historical --resume  # Resume any incomplete session
    """
    setup_logging(verbose)

    # Validate options
    option_count = sum([year is not None, all_years, start is not None])
    if option_count == 0:
        console.print("[red]Error: Must specify --year, --all, or --start/--end[/red]")
        raise typer.Exit(1)
    if option_count > 1 and not (start and end and not year and not all_years):
        console.print("[red]Error: Use only one of --year, --all, or --start/--end[/red]")
        raise typer.Exit(1)

    if start and not end:
        console.print("[red]Error: --start requires --end[/red]")
        raise typer.Exit(1)
    if end and not start:
        console.print("[red]Error: --end requires --start[/red]")
        raise typer.Exit(1)

    console.print("\n[bold blue]MCF Historical Scraper[/bold blue]")

    if dry_run:
        console.print("[yellow]DRY RUN[/yellow] - No data will be fetched")

    if year:
        console.print(f"Year: [green]{year}[/green]")
        estimated = YEAR_ESTIMATES.get(year, 1_000_000)
        console.print(f"Estimated jobs: ~{estimated:,}")
    elif all_years:
        console.print("Scraping: [green]All years (2019-2026)[/green]")
        total_estimated = sum(YEAR_ESTIMATES.values())
        console.print(f"Total estimated jobs: ~{total_estimated:,}")
    elif start and end:
        console.print(f"Range: [green]{start}[/green] to [green]{end}[/green]")

    console.print(f"Rate limit: {rate_limit} req/sec")
    console.print(f"Database: {db_path}")
    console.print()

    async def run():
        async with HistoricalScraper(
            db_path=db_path,
            requests_per_second=rate_limit,
            not_found_threshold=not_found_threshold,
        ) as scraper:
            # Set up progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.fields[stats]}"),
                console=console,
                transient=True,
            ) as progress:
                task_id = None

                async def update_progress(p):
                    nonlocal task_id
                    if task_id is None:
                        total = p.end_seq - p.start_seq + 1 if p.end_seq else None
                        task_id = progress.add_task(
                            f"Year {p.year}",
                            total=total,
                            stats="",
                        )
                    completed = p.current_seq - p.start_seq
                    stats = f"Found: {p.jobs_found:,} | Not found: {p.jobs_not_found:,}"
                    progress.update(task_id, completed=completed, stats=stats)

                if year:
                    result = await scraper.scrape_year(
                        year,
                        resume=resume,
                        progress_callback=update_progress,
                        dry_run=dry_run,
                    )
                    results = {year: result}

                elif all_years:
                    results = {}
                    for y in sorted(YEAR_ESTIMATES.keys()):
                        task_id = None  # Reset for new year
                        results[y] = await scraper.scrape_year(
                            y,
                            resume=resume,
                            progress_callback=update_progress,
                            dry_run=dry_run,
                        )

                elif start and end:
                    result = await scraper.scrape_range(
                        start,
                        end,
                        progress_callback=update_progress,
                        dry_run=dry_run,
                    )
                    results = {result.year: result}

        # Display results
        console.print("\n[bold green]Scrape Complete![/bold green]\n")

        table = Table(title="Results by Year", show_header=True)
        table.add_column("Year", style="cyan")
        table.add_column("Jobs Found", justify="right", style="green")
        table.add_column("Not Found", justify="right", style="dim")
        table.add_column("Success Rate", justify="right")
        table.add_column("Last Seq", justify="right")

        total_found = 0
        total_not_found = 0

        for y, p in sorted(results.items()):
            total_found += p.jobs_found
            total_not_found += p.jobs_not_found

            table.add_row(
                str(y),
                f"{p.jobs_found:,}",
                f"{p.jobs_not_found:,}",
                f"{p.success_rate:.1f}%",
                f"{p.current_seq:,}",
            )

        console.print(table)

        console.print(f"\n[bold]Total jobs found:[/bold] [green]{total_found:,}[/green]")
        console.print(f"[bold]Total not found:[/bold] {total_not_found:,}")

        if not dry_run:
            # Show current database stats
            db = MCFDatabase(db_path)
            total_jobs = db.count_jobs()
            console.print(f"\n[bold]Total jobs in database:[/bold] [cyan]{total_jobs:,}[/cyan]")

    asyncio.run(run())


@app.command(name="historical-status")
def historical_status(
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
) -> None:
    """
    Show status of historical scrape sessions.

    Displays progress for each year being scraped, including jobs found,
    not found, and current sequence position.
    """
    db = MCFDatabase(db_path)

    # Get sessions
    sessions = db.get_all_historical_sessions()

    if not sessions:
        console.print("No historical scrape sessions found")
        console.print("\nRun [bold]mcf scrape-historical --year 2023[/bold] to start scraping")
        return

    console.print("\n[bold blue]Historical Scrape Status[/bold blue]\n")

    # Active sessions
    active = [s for s in sessions if s["status"] == "in_progress"]
    if active:
        console.print("[bold]Active Sessions:[/bold]\n")

        table = Table(show_header=True)
        table.add_column("ID")
        table.add_column("Year", style="cyan")
        table.add_column("Progress", justify="right")
        table.add_column("Found", justify="right", style="green")
        table.add_column("Not Found", justify="right")
        table.add_column("Consecutive NF", justify="right", style="dim")
        table.add_column("Started")

        for s in active:
            end_seq = s["end_seq"] or YEAR_ESTIMATES.get(s["year"], 1_000_000)
            progress_pct = (s["current_seq"] - s["start_seq"]) / (end_seq - s["start_seq"]) * 100
            progress_str = f"{s['current_seq']:,}/{end_seq:,} ({progress_pct:.1f}%)"

            table.add_row(
                str(s["id"]),
                str(s["year"]),
                progress_str,
                f"{s['jobs_found']:,}",
                f"{s['jobs_not_found']:,}",
                str(s["consecutive_not_found"]),
                str(s["started_at"])[:16] if s["started_at"] else "",
            )

        console.print(table)
        console.print()

    # Completed sessions
    completed = [s for s in sessions if s["status"] == "completed"]
    if completed:
        console.print("[bold]Completed Sessions:[/bold]\n")

        table = Table(show_header=True)
        table.add_column("Year", style="cyan")
        table.add_column("Jobs Found", justify="right", style="green")
        table.add_column("Max Seq", justify="right")
        table.add_column("Completed")

        for s in completed:
            table.add_row(
                str(s["year"]),
                f"{s['jobs_found']:,}",
                f"{s['current_seq']:,}",
                str(s["completed_at"])[:16] if s["completed_at"] else "",
            )

        console.print(table)
        console.print()

    # Summary stats
    stats = db.get_historical_stats()

    if stats.get("jobs_by_year"):
        console.print("[bold]Jobs in Database by Year:[/bold]\n")

        table = Table(show_header=True)
        table.add_column("Year", style="cyan")
        table.add_column("Jobs", justify="right", style="green")
        table.add_column("Est. Total", justify="right", style="dim")
        table.add_column("Coverage", justify="right")

        for year, count in sorted(stats["jobs_by_year"].items(), reverse=True):
            estimated = YEAR_ESTIMATES.get(int(year), 0)
            coverage = (count / estimated * 100) if estimated else 0
            table.add_row(
                year,
                f"{count:,}",
                f"~{estimated:,}" if estimated else "?",
                f"{coverage:.1f}%",
            )

        console.print(table)

    # Instructions
    if active:
        console.print("\nResume with: [bold]mcf scrape-historical --year YEAR[/bold]")


# Daemon commands


@app.command(name="daemon")
def daemon_cmd(
    action: str = typer.Argument(..., help="Action: start, stop, or status"),
    year: Optional[int] = typer.Option(
        None, "--year", "-y", help="Year to scrape (for start action)"
    ),
    all_years: bool = typer.Option(
        False, "--all", help="Scrape all years (for start action)"
    ),
    rate_limit: float = typer.Option(
        2.0, "--rate-limit", "-r", help="Initial requests per second"
    ),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """
    Manage the background scraper daemon.

    The daemon runs historical scraping in the background, surviving terminal
    closure and detecting sleep/wake cycles.

    Examples:
        mcf daemon start --year 2023     # Start scraping 2023 in background
        mcf daemon start --all           # Start scraping all years
        mcf daemon status                # Check daemon status
        mcf daemon stop                  # Stop the daemon
    """
    setup_logging(verbose)

    db = MCFDatabase(db_path)
    daemon = ScraperDaemon(db)

    if action == "status":
        status = daemon.status()

        console.print("\n[bold blue]Daemon Status[/bold blue]\n")

        if status["running"]:
            console.print(f"[green]● Running[/green] (PID {status['pid']})")
        else:
            console.print("[dim]○ Stopped[/dim]")

        console.print(f"PID file: {status['pidfile']}")
        console.print(f"Log file: {status['logfile']}")

        if status.get("last_heartbeat"):
            console.print(f"\nLast heartbeat: {status['last_heartbeat']}")

        if status.get("current_year"):
            console.print(f"Current year: {status['current_year']}")
            if status.get("current_seq"):
                console.print(f"Current sequence: {status['current_seq']:,}")

        if status.get("started_at"):
            console.print(f"Started at: {status['started_at']}")

    elif action == "start":
        if not year and not all_years:
            console.print("[red]Error: Must specify --year or --all for start action[/red]")
            raise typer.Exit(1)

        try:
            console.print("\n[bold blue]Starting Daemon[/bold blue]")

            if year:
                console.print(f"Year: [green]{year}[/green]")
            elif all_years:
                console.print("Mode: [green]All years (2019-2026)[/green]")

            console.print(f"Rate limit: {rate_limit} req/sec")
            console.print(f"Database: {db_path}")
            console.print()

            # Define the scraper function to run in daemon
            async def run_scraper():
                async with HistoricalScraper(
                    db_path=db_path,
                    requests_per_second=rate_limit,
                ) as scraper:
                    if year:
                        return await scraper.scrape_year(year, resume=True)
                    else:
                        return await scraper.scrape_all_years(resume=True)

            pid = daemon.start(run_scraper)
            console.print(f"[green]Daemon started with PID {pid}[/green]")
            console.print(f"\nLogs: [cyan]{daemon.logfile}[/cyan]")
            console.print("\nCheck status: [bold]mcf daemon status[/bold]")
            console.print("Stop daemon: [bold]mcf daemon stop[/bold]")

        except DaemonAlreadyRunning as e:
            console.print(f"[yellow]{e}[/yellow]")
            console.print("Use [bold]mcf daemon stop[/bold] first")
            raise typer.Exit(1)

    elif action == "stop":
        try:
            console.print("Stopping daemon...")
            daemon.stop()
            console.print("[green]Daemon stopped[/green]")
        except DaemonNotRunning:
            console.print("[yellow]No daemon is running[/yellow]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: start, stop, status")
        raise typer.Exit(1)


# Gap analysis commands


@app.command(name="gaps")
def show_gaps(
    year: Optional[int] = typer.Option(
        None, "--year", "-y", help="Specific year to check"
    ),
    all_years: bool = typer.Option(
        False, "--all", help="Check all years"
    ),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
) -> None:
    """
    Show gaps in scraped job sequences.

    Analyzes fetch_attempts to find missing sequence ranges that need
    to be retried.

    Examples:
        mcf gaps --year 2023     # Check gaps for 2023
        mcf gaps --all           # Check all years
    """
    db = MCFDatabase(db_path)

    if not year and not all_years:
        console.print("[red]Error: Must specify --year or --all[/red]")
        raise typer.Exit(1)

    console.print("\n[bold blue]Gap Analysis[/bold blue]\n")

    years_to_check = [year] if year else sorted(YEAR_ESTIMATES.keys())

    total_gaps = 0
    total_missing = 0
    total_errors = 0

    for y in years_to_check:
        gaps = db.get_missing_sequences(y)
        failed = db.get_failed_attempts(y)
        stats = db.get_attempt_stats(y)

        if not stats.get("total"):
            if year:  # Only show message if specific year requested
                console.print(f"[yellow]No fetch attempts recorded for year {y}[/yellow]")
            continue

        missing_count = sum(end - start + 1 for start, end in gaps)
        total_gaps += len(gaps)
        total_missing += missing_count
        total_errors += len(failed)

        console.print(f"[bold]Year {y}:[/bold]")
        console.print(f"  Sequences attempted: {stats.get('total', 0):,}")
        console.print(f"  Found: [green]{stats.get('found', 0):,}[/green]")
        console.print(f"  Not found: {stats.get('not_found', 0):,}")
        console.print(f"  Skipped: {stats.get('skipped', 0):,}")
        console.print(f"  Errors: [red]{stats.get('error', 0):,}[/red]")

        if gaps:
            console.print(f"\n  [yellow]Gaps ({len(gaps)} ranges, {missing_count:,} sequences):[/yellow]")
            for start, end in gaps[:5]:  # Show first 5 gaps
                console.print(f"    {start:,} - {end:,} ({end - start + 1:,} missing)")
            if len(gaps) > 5:
                console.print(f"    ... and {len(gaps) - 5} more gaps")
        else:
            console.print("  [green]No gaps detected[/green]")

        if failed:
            console.print(f"\n  [red]Failed attempts: {len(failed):,}[/red]")

        console.print()

    # Summary
    if all_years:
        console.print("[bold]Summary:[/bold]")
        console.print(f"  Total gaps: {total_gaps}")
        console.print(f"  Total missing sequences: {total_missing:,}")
        console.print(f"  Total errors to retry: {total_errors:,}")

        if total_missing + total_errors > 0:
            console.print(f"\nRun [bold]mcf retry-gaps --all[/bold] to retry missing sequences")


@app.command(name="retry-gaps")
def retry_gaps(
    year: Optional[int] = typer.Option(
        None, "--year", "-y", help="Specific year to retry"
    ),
    all_years: bool = typer.Option(
        False, "--all", help="Retry all years"
    ),
    rate_limit: float = typer.Option(
        2.0, "--rate-limit", "-r", help="Requests per second"
    ),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """
    Retry fetching jobs for missing/failed sequences.

    Finds gaps in fetch_attempts and retries each sequence.

    Examples:
        mcf retry-gaps --year 2023     # Retry gaps for 2023
        mcf retry-gaps --all           # Retry all years
    """
    setup_logging(verbose)

    if not year and not all_years:
        console.print("[red]Error: Must specify --year or --all[/red]")
        raise typer.Exit(1)

    console.print("\n[bold blue]Retrying Gaps[/bold blue]\n")

    async def run():
        async with HistoricalScraper(
            db_path=db_path,
            requests_per_second=rate_limit,
        ) as scraper:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.fields[stats]}"),
                console=console,
            ) as progress:
                task_id = None

                async def update_progress(p):
                    nonlocal task_id
                    if task_id is None:
                        task_id = progress.add_task(
                            f"Year {p.year}",
                            total=p.end_seq - p.start_seq + 1 if p.end_seq else None,
                            stats="",
                        )
                    completed = p.current_seq - p.start_seq if p.start_seq else 0
                    stats = f"Recovered: {p.jobs_found:,}"
                    progress.update(task_id, completed=completed, stats=stats)

                if year:
                    result = await scraper.retry_gaps(year, update_progress)
                    results = {year: result}
                else:
                    results = {}
                    for y in sorted(YEAR_ESTIMATES.keys()):
                        task_id = None
                        results[y] = await scraper.retry_gaps(y, update_progress)

        # Display results
        console.print("\n[bold green]Gap Retry Complete![/bold green]\n")

        table = Table(title="Results by Year", show_header=True)
        table.add_column("Year", style="cyan")
        table.add_column("Recovered", justify="right", style="green")
        table.add_column("Still Missing", justify="right", style="dim")

        total_recovered = 0
        total_missing = 0

        for y, p in sorted(results.items()):
            if p.jobs_found > 0 or p.jobs_not_found > 0:
                total_recovered += p.jobs_found
                total_missing += p.jobs_not_found

                table.add_row(
                    str(y),
                    f"{p.jobs_found:,}",
                    f"{p.jobs_not_found:,}",
                )

        if total_recovered > 0 or total_missing > 0:
            console.print(table)
            console.print(f"\n[bold]Total recovered:[/bold] [green]{total_recovered:,}[/green]")
        else:
            console.print("[green]No gaps to retry![/green]")

    asyncio.run(run())


@app.command(name="attempt-stats")
def attempt_stats(
    year: Optional[int] = typer.Option(
        None, "--year", "-y", help="Specific year to show"
    ),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
) -> None:
    """
    Show fetch attempt statistics.

    Displays counts of found, not_found, error, and skipped attempts
    for tracking scraper completeness.

    Examples:
        mcf attempt-stats              # All years summary
        mcf attempt-stats --year 2023  # Specific year details
    """
    db = MCFDatabase(db_path)

    console.print("\n[bold blue]Fetch Attempt Statistics[/bold blue]\n")

    if year:
        stats = db.get_attempt_stats(year)

        if not stats.get("total"):
            console.print(f"[yellow]No fetch attempts recorded for year {year}[/yellow]")
            return

        console.print(f"[bold]Year {year}:[/bold]")
        console.print(f"  Total attempts: {stats['total']:,}")
        console.print(f"  Found: [green]{stats.get('found', 0):,}[/green]")
        console.print(f"  Not found: {stats.get('not_found', 0):,}")
        console.print(f"  Skipped: {stats.get('skipped', 0):,}")
        console.print(f"  Errors: [red]{stats.get('error', 0):,}[/red]")

        if stats.get("min_sequence"):
            console.print(f"\n  Sequence range: {stats['min_sequence']:,} - {stats['max_sequence']:,}")
            console.print(f"  Range size: {stats['sequence_range']:,}")

            coverage = stats['total'] / stats['sequence_range'] * 100
            console.print(f"  Coverage: {coverage:.1f}%")

    else:
        all_stats = db.get_all_attempt_stats()

        if not all_stats:
            console.print("[yellow]No fetch attempts recorded[/yellow]")
            return

        table = Table(show_header=True)
        table.add_column("Year", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Found", justify="right", style="green")
        table.add_column("Not Found", justify="right")
        table.add_column("Skipped", justify="right", style="dim")
        table.add_column("Errors", justify="right", style="red")

        grand_total = 0
        grand_found = 0

        for y in sorted(all_stats.keys()):
            stats = all_stats[y]
            grand_total += stats.get("total", 0)
            grand_found += stats.get("found", 0)

            table.add_row(
                str(y),
                f"{stats.get('total', 0):,}",
                f"{stats.get('found', 0):,}",
                f"{stats.get('not_found', 0):,}",
                f"{stats.get('skipped', 0):,}",
                f"{stats.get('error', 0):,}",
            )

        console.print(table)
        console.print(f"\n[bold]Grand total:[/bold] {grand_total:,} attempts, {grand_found:,} jobs found")


# Embedding commands


@app.command(name="embed-generate")
def generate_embeddings(
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Jobs to process in each batch"
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip jobs that already have embeddings",
    ),
    build_index: bool = typer.Option(
        True,
        "--build-index/--no-build-index",
        help="Build FAISS indexes after embedding generation",
    ),
    index_dir: str = typer.Option(
        "data/embeddings", "--index-dir", help="Directory for FAISS index files"
    ),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Path to SQLite database"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Generate embeddings for all jobs and build FAISS indexes.

    This preprocesses all jobs to create semantic embeddings for similarity search,
    then builds FAISS indexes for efficient nearest-neighbor lookup.

    Examples:
        mcf embed-generate
        mcf embed-generate --batch-size 64
        mcf embed-generate --no-skip-existing  # Regenerate all
        mcf embed-generate --no-build-index    # Skip index building
    """
    setup_logging(verbose)

    console.print("\n[bold blue]Generating Embeddings[/bold blue]")
    console.print("━" * 40)

    db = MCFDatabase(db_path)
    generator = EmbeddingGenerator()

    console.print(f"Model: [green]{generator.model_name}[/green] ({generator.DIMENSION} dimensions)")
    console.print(f"Database: [green]{db_path}[/green]")
    console.print()

    # Get initial count for progress
    if skip_existing:
        jobs_to_process = db.get_jobs_without_embeddings(limit=1000000)
        total_jobs = len(jobs_to_process)
        if total_jobs == 0:
            console.print("[green]✓ All jobs already have embeddings![/green]")
            # Still build index if requested and we have embeddings
            if build_index:
                emb_stats = db.get_embedding_stats()
                if emb_stats["job_embeddings"] > 0:
                    _build_faiss_indexes(db, generator, index_dir, console)
            return
        console.print(f"Jobs to process: [cyan]{total_jobs:,}[/cyan] (skipping existing)")
    else:
        total_jobs = db.count_jobs()
        console.print(f"Jobs to process: [cyan]{total_jobs:,}[/cyan] (regenerating all)")

    console.print()

    # Phase 1: Generate embeddings
    console.print("[bold]Phase 1: Generating Embeddings[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed:,}/{task.total:,}"),
        TextColumn("[dim]({task.fields[speed]:.1f} jobs/sec)"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Jobs", total=total_jobs, speed=0.0)

        def on_progress(stats: EmbeddingStats):
            progress.update(
                task_id,
                completed=stats.jobs_processed,
                speed=stats.jobs_per_second,
            )

        stats = generator.generate_all(
            db,
            batch_size=batch_size,
            skip_existing=skip_existing,
            progress_callback=on_progress,
        )

    # Print embedding summary
    console.print("\n[bold green]✓ Embedding generation complete![/bold green]\n")

    table = Table(title="Embedding Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Jobs processed", f"{stats.jobs_processed:,}")
    table.add_row("Jobs skipped", f"{stats.jobs_skipped:,}")
    if stats.jobs_failed > 0:
        table.add_row("Jobs failed", f"[red]{stats.jobs_failed:,}[/red]")
    table.add_row("Skills extracted", f"{stats.unique_skills:,}")
    table.add_row("Skill clusters", f"{stats.skill_clusters:,}")

    # Format elapsed time
    elapsed = stats.elapsed_seconds
    if elapsed >= 60:
        minutes, seconds = divmod(int(elapsed), 60)
        elapsed_str = f"{minutes}m {seconds}s"
    else:
        elapsed_str = f"{elapsed:.1f}s"
    table.add_row("Total time", elapsed_str)
    table.add_row("Speed", f"{stats.jobs_per_second:.1f} jobs/sec")

    console.print(table)

    # Phase 2: Build FAISS indexes
    if not build_index:
        console.print("\n[yellow]Skipping index building (--no-build-index)[/yellow]")
        return

    _build_faiss_indexes(db, generator, index_dir, console)


def _build_faiss_indexes(
    db: MCFDatabase,
    generator: EmbeddingGenerator,
    index_dir: str,
    console: Console,
) -> None:
    """Build FAISS indexes from embeddings in database."""
    import numpy as np

    console.print("\n[bold]Phase 2: Building FAISS Indexes[/bold]\n")

    index_path = Path(index_dir)
    index_manager = FAISSIndexManager(
        index_dir=index_path,
        model_version=generator.model_name,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load job embeddings from database
        task = progress.add_task("Loading job embeddings from database...", total=None)
        job_uuids, job_embeddings = db.get_all_embeddings("job")

        if len(job_uuids) == 0:
            progress.update(task, description="[yellow]No job embeddings found[/yellow]")
            console.print("\n[yellow]No job embeddings found. Run embed-generate first.[/yellow]")
            return

        # Build job index
        progress.update(task, description=f"Building job index ({len(job_uuids):,} vectors)...")
        index_manager.build_job_index(job_embeddings, job_uuids)

        # Load and build skill index
        progress.update(task, description="Loading skill embeddings...")
        skill_names, skill_embeddings = db.get_all_embeddings("skill")

        if len(skill_names) > 0:
            progress.update(task, description=f"Building skill index ({len(skill_names):,} skills)...")
            index_manager.build_skill_index(skill_embeddings, skill_names)

        # Generate company centroids and build company index
        progress.update(task, description="Generating company centroids...")
        company_centroids = generator.generate_company_centroids_from_db(db)
        if company_centroids:
            total_centroids = sum(len(c) for c in company_centroids.values())
            progress.update(
                task,
                description=f"Building company index ({len(company_centroids):,} companies, {total_centroids:,} centroids)...",
            )
            index_manager.build_company_index(company_centroids)

        # Save indexes to disk
        progress.update(task, description="Saving indexes to disk...")
        index_manager.save()

        progress.update(task, description="[green]✓ Complete[/green]")

    # Print index summary
    console.print("\n[bold green]✓ FAISS indexes built![/bold green]\n")

    index_stats = index_manager.get_stats()
    table = Table(title="Index Summary", show_header=True)
    table.add_column("Index", style="cyan")
    table.add_column("Vectors", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Type")

    if "jobs" in index_stats["indexes"]:
        job_idx = index_stats["indexes"]["jobs"]
        table.add_row(
            "jobs.index",
            f"{job_idx['total_vectors']:,}",
            f"{job_idx['estimated_memory_mb']:.1f} MB",
            job_idx["index_type"],
        )

    if "skills" in index_stats["indexes"]:
        skill_idx = index_stats["indexes"]["skills"]
        table.add_row(
            "skills.index",
            f"{skill_idx['total_skills']:,}",
            "-",
            skill_idx["index_type"],
        )

    if "companies" in index_stats["indexes"]:
        co_idx = index_stats["indexes"]["companies"]
        table.add_row(
            "companies.index",
            f"{co_idx['total_companies']:,} ({co_idx['total_centroids']:,} centroids)",
            "-",
            co_idx["index_type"],
        )

    console.print(table)
    console.print(f"\nIndexes saved to: [cyan]{index_dir}[/cyan]")


@app.command(name="embed-sync")
def sync_embeddings(
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Jobs to process in each batch"
    ),
    update_index: bool = typer.Option(
        True,
        "--update-index/--no-update-index",
        help="Update FAISS indexes with new embeddings",
    ),
    index_dir: str = typer.Option(
        "data/embeddings", "--index-dir", help="Directory for FAISS index files"
    ),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Path to SQLite database"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Generate embeddings for new jobs and update indexes.

    Only processes jobs that don't have embeddings yet, then adds
    them to existing FAISS indexes for incremental updates.

    Examples:
        mcf embed-sync
        mcf scrape "data scientist" && mcf embed-sync  # Chain commands
        mcf embed-sync --no-update-index  # Skip index update
    """
    setup_logging(verbose)

    console.print("\n[bold blue]Syncing Embeddings[/bold blue]")
    console.print("━" * 40)

    db = MCFDatabase(db_path)
    generator = EmbeddingGenerator()

    # Check how many jobs need embeddings
    jobs_to_process = db.get_jobs_without_embeddings(limit=1000000)
    total_new = len(jobs_to_process)

    if total_new == 0:
        console.print("[green]✓ All jobs are up-to-date![/green]")
        return

    console.print(f"New jobs to embed: [cyan]{total_new:,}[/cyan]")
    console.print()

    # Collect new UUIDs for index update
    new_uuids = [job["uuid"] for job in jobs_to_process]

    # Run with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed:,}/{task.total:,}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Embedding", total=total_new)

        def on_progress(stats: EmbeddingStats):
            progress.update(task_id, completed=stats.jobs_processed)

        stats = generator.generate_all(
            db,
            batch_size=batch_size,
            skip_existing=True,
            progress_callback=on_progress,
        )

    console.print(f"\n[green]✓ Synced {stats.jobs_processed:,} jobs in {stats.elapsed_seconds:.1f}s[/green]")

    # Update FAISS index if requested
    if not update_index:
        console.print("[yellow]Skipping index update (--no-update-index)[/yellow]")
        return

    _update_faiss_index(db, generator, index_dir, new_uuids, console)


def _update_faiss_index(
    db: MCFDatabase,
    generator: EmbeddingGenerator,
    index_dir: str,
    new_uuids: list[str],
    console: Console,
) -> None:
    """Add new embeddings to existing FAISS index."""
    import numpy as np

    index_path = Path(index_dir)
    index_manager = FAISSIndexManager(
        index_dir=index_path,
        model_version=generator.model_name,
    )

    # Try to load existing index
    if not index_manager.exists():
        console.print("\n[yellow]No existing FAISS index found.[/yellow]")
        console.print("Run [bold]mcf embed-generate[/bold] to build indexes.")
        return

    console.print("\n[bold]Updating FAISS Index[/bold]")

    try:
        index_manager.load()
    except IndexCompatibilityError as e:
        console.print(f"\n[red]Index compatibility error: {e}[/red]")
        console.print("Run [bold]mcf embed-generate --no-skip-existing[/bold] to rebuild.")
        return

    # Get embeddings for new jobs
    embeddings_dict = db.get_embeddings_for_uuids(new_uuids)

    if not embeddings_dict:
        console.print("[yellow]No new embeddings to add to index[/yellow]")
        return

    # Prepare arrays for add_jobs
    uuids = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[uuid] for uuid in uuids], dtype=np.float32)

    # Add to index
    index_manager.add_jobs(embeddings, uuids)
    index_manager.save()

    console.print(f"[green]✓ Added {len(uuids):,} jobs to FAISS index[/green]")
    console.print(f"Total jobs in index: [cyan]{index_manager.indexes['jobs'].ntotal:,}[/cyan]")


@app.command(name="embed-status")
def embedding_status(
    index_dir: str = typer.Option(
        "data/embeddings", "--index-dir", help="Directory for FAISS index files"
    ),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Path to SQLite database"),
) -> None:
    """
    Show embedding and FAISS index status.

    Displays statistics about embeddings in the database and FAISS indexes on disk.

    Example:
        mcf embed-status
    """
    db = MCFDatabase(db_path)
    stats = db.get_embedding_stats()

    console.print("\n[bold blue]Embedding Status[/bold blue]")
    console.print("━" * 40)

    # Jobs section
    console.print("\n[bold]Jobs:[/bold]")
    console.print(f"  Total in database:     {stats['total_jobs']:,}")
    console.print(f"  With embeddings:       {stats['job_embeddings']:,}")

    coverage = stats["coverage_pct"]
    coverage_style = "green" if coverage >= 95 else "yellow" if coverage >= 80 else "red"
    console.print(f"  Coverage:              [{coverage_style}]{coverage:.1f}%[/{coverage_style}]")

    if stats["model_version"]:
        console.print(f"  Model version:         {stats['model_version']}")

    # Skills section
    console.print("\n[bold]Skills:[/bold]")
    console.print(f"  With embeddings:       {stats['skill_embeddings']:,}")

    # Companies section (if implemented)
    if stats.get("company_embeddings", 0) > 0:
        console.print("\n[bold]Companies:[/bold]")
        console.print(f"  With embeddings:       {stats['company_embeddings']:,}")

    # FAISS Index section
    console.print("\n[bold]FAISS Indexes:[/bold]")

    index_path = Path(index_dir)
    index_manager = FAISSIndexManager(
        index_dir=index_path,
        model_version=stats.get("model_version") or "all-MiniLM-L6-v2",
    )

    if index_manager.exists():
        try:
            index_manager.load()
            index_stats = index_manager.get_stats()

            table = Table(show_header=True, box=None)
            table.add_column("Index", style="cyan")
            table.add_column("Vectors", justify="right")
            table.add_column("Memory", justify="right")
            table.add_column("Status")

            # Job index
            if "jobs" in index_stats["indexes"]:
                job_idx = index_stats["indexes"]["jobs"]
                mem_str = f"{job_idx['estimated_memory_mb']:.1f} MB"
                # Check if index is in sync with embeddings
                in_sync = job_idx["total_vectors"] == stats["job_embeddings"]
                status = "[green]✓ ready[/green]" if in_sync else f"[yellow]⚠ {stats['job_embeddings'] - job_idx['total_vectors']:+,} out of sync[/yellow]"
                table.add_row("jobs.index", f"{job_idx['total_vectors']:,}", mem_str, status)

            # Skill index
            if "skills" in index_stats["indexes"]:
                skill_idx = index_stats["indexes"]["skills"]
                in_sync = skill_idx["total_skills"] == stats["skill_embeddings"]
                status = "[green]✓ ready[/green]" if in_sync else "[yellow]⚠ out of sync[/yellow]"
                table.add_row("skills.index", f"{skill_idx['total_skills']:,}", "-", status)

            # Company index
            if "companies" in index_stats["indexes"]:
                company_idx = index_stats["indexes"]["companies"]
                table.add_row(
                    "companies.index",
                    f"{company_idx['total_centroids']:,} centroids",
                    "-",
                    "[green]✓ ready[/green]",
                )

            console.print(table)
            console.print(f"\n  Index directory: [dim]{index_dir}[/dim]")

        except IndexCompatibilityError as e:
            console.print(f"  [red]⚠ Index incompatible: {e}[/red]")
            console.print("  Run [bold]mcf embed-generate --no-skip-existing[/bold] to rebuild")
    else:
        console.print("  [yellow]No FAISS indexes found[/yellow]")
        console.print("  Run [bold]mcf embed-generate[/bold] to build indexes")

    # Check for cluster files
    cluster_dir = Path(index_dir)
    cluster_files = [
        "skill_clusters.pkl",
        "skill_to_cluster.pkl",
        "skill_cluster_centroids.pkl",
    ]

    console.print("\n[bold]Cluster Files:[/bold]")
    for filename in cluster_files:
        filepath = cluster_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            if size >= 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size >= 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            console.print(f"  {filename:<30} [green]✓ exists[/green] ({size_str})")
        else:
            console.print(f"  {filename:<30} [dim]not found[/dim]")

    # Missing embeddings warning
    if coverage < 100:
        missing = stats["total_jobs"] - stats["job_embeddings"]
        console.print(f"\n[yellow]⚠ {missing:,} jobs missing embeddings[/yellow]")
        console.print("Run [bold]mcf embed-sync[/bold] to generate missing embeddings")


@app.command(name="embed-upgrade")
def upgrade_embeddings(
    model: str = typer.Argument(..., help="New model name (e.g., all-mpnet-base-v2)"),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Jobs to process in each batch"
    ),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Path to SQLite database"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    confirm: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt"
    ),
) -> None:
    """
    Re-generate all embeddings with a new model version.

    This will:
    1. Delete all existing embeddings
    2. Generate new embeddings with the specified model
    3. Rebuild skill clusters

    Use this when upgrading to a better embedding model.

    Examples:
        mcf embed-upgrade all-mpnet-base-v2
        mcf embed-upgrade all-MiniLM-L12-v2 --yes
    """
    setup_logging(verbose)

    db = MCFDatabase(db_path)
    stats = db.get_embedding_stats()

    current_model = stats.get("model_version") or "none"
    total_jobs = stats["total_jobs"]
    current_embeddings = stats["job_embeddings"]

    # Show warning
    console.print("\n[bold yellow]⚠️  This will regenerate ALL embeddings![/bold yellow]\n")
    console.print(f"Current model: [cyan]{current_model}[/cyan]")
    console.print(f"New model:     [green]{model}[/green]")
    console.print(f"Jobs to re-embed: [cyan]{total_jobs:,}[/cyan]")

    if current_embeddings > 0:
        console.print(f"Embeddings to delete: [red]{current_embeddings:,}[/red]")

    # Confirm
    if not confirm:
        console.print()
        proceed = typer.confirm("Continue?", default=False)
        if not proceed:
            console.print("[dim]Aborted[/dim]")
            raise typer.Exit(0)

    console.print()

    # Step 1: Delete existing embeddings
    if current_embeddings > 0:
        console.print("Deleting existing embeddings...")
        if current_model and current_model != "none":
            deleted = db.delete_embeddings_for_model(current_model)
            console.print(f"  Deleted {deleted:,} embeddings for model '{current_model}'")

        # Also delete any embeddings with different/no model version
        with db._connection() as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            if remaining > 0:
                conn.execute("DELETE FROM embeddings")
                console.print(f"  Deleted {remaining:,} remaining embeddings")

    # Step 2: Generate new embeddings with new model
    console.print(f"\nGenerating embeddings with [green]{model}[/green]...")
    console.print()

    generator = EmbeddingGenerator(model_name=model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed:,}/{task.total:,}"),
        TextColumn("[dim]({task.fields[speed]:.1f} jobs/sec)"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Jobs", total=total_jobs, speed=0.0)

        def on_progress(emb_stats: EmbeddingStats):
            progress.update(
                task_id,
                completed=emb_stats.jobs_processed,
                speed=emb_stats.jobs_per_second,
            )

        final_stats = generator.generate_all(
            db,
            batch_size=batch_size,
            skip_existing=False,  # Regenerate all
            progress_callback=on_progress,
        )

    # Summary
    console.print("\n[bold green]✓ Embedding upgrade complete![/bold green]\n")

    table = Table(title="Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("New model", model)
    table.add_row("Jobs processed", f"{final_stats.jobs_processed:,}")
    table.add_row("Skills clustered", f"{final_stats.unique_skills:,}")

    elapsed = final_stats.elapsed_seconds
    if elapsed >= 60:
        minutes, seconds = divmod(int(elapsed), 60)
        elapsed_str = f"{minutes}m {seconds}s"
    else:
        elapsed_str = f"{elapsed:.1f}s"
    table.add_row("Total time", elapsed_str)

    console.print(table)


# Semantic search command


@app.command(name="search-semantic")
def semantic_search_cli(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    salary_min: Optional[int] = typer.Option(None, "--salary-min", help="Minimum salary"),
    salary_max: Optional[int] = typer.Option(None, "--salary-max", help="Maximum salary"),
    company: Optional[str] = typer.Option(None, "--company", "-c", help="Filter by company"),
    employment_type: Optional[str] = typer.Option(
        None, "--employment-type", "-e", help="Filter by employment type"
    ),
    region: Optional[str] = typer.Option(None, "--region", "-r", help="Filter by region"),
    alpha: float = typer.Option(
        0.7, "--alpha", help="Semantic vs keyword weight (0=keyword, 1=semantic)"
    ),
    no_expand: bool = typer.Option(False, "--no-expand", help="Disable query expansion"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    db_path: str = typer.Option("data/mcf_jobs.db", "--db", help="Database path"),
    index_dir: str = typer.Option("data/embeddings", "--index-dir", help="FAISS index directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """
    Semantic search for jobs.

    Combines keyword matching with semantic similarity for better results
    than plain text search. Requires embeddings to be generated first
    (run 'embed-generate' if not done yet).

    Examples:
        mcf search-semantic "machine learning engineer"
        mcf search-semantic "python developer" --salary-min 8000
        mcf search-semantic "data scientist" --company Google
        mcf search-semantic "ML" --no-expand
        mcf search-semantic "AI engineer" --json
    """
    setup_logging(verbose)

    engine = SemanticSearchEngine(db_path, Path(index_dir))

    console.print("[dim]Loading search indexes...[/dim]")
    try:
        loaded = engine.load()
    except Exception as e:
        console.print(f"[red]Failed to load indexes: {e}[/red]")
        console.print("[yellow]Run 'mcf embed-generate' first to build indexes.[/yellow]")
        raise typer.Exit(1)

    if not loaded:
        console.print("[yellow]Running in degraded mode (keyword search only)[/yellow]")

    request = SearchRequest(
        query=query,
        limit=limit,
        salary_min=salary_min,
        salary_max=salary_max,
        company=company,
        employment_type=employment_type,
        region=region,
        alpha=alpha,
        expand_query=not no_expand,
    )

    response = engine.search(request)

    if json_output:
        _print_search_json(response, query)
    else:
        _display_search_results(response, query)


def _print_search_json(response: SearchResponse, query: str) -> None:
    """Print search results as JSON."""
    import json
    from dataclasses import asdict

    data = asdict(response)
    data["query"] = query

    # Convert date objects to strings for JSON serialization
    for result in data.get("results", []):
        if result.get("posted_date"):
            result["posted_date"] = str(result["posted_date"])

    console.print(json.dumps(data, indent=2))


def _display_search_results(response: SearchResponse, query: str) -> None:
    """Display search results as a Rich table."""
    console.print(f"\n[bold blue]Semantic Search Results[/bold blue]")
    console.print("━" * 70)

    # Query info
    console.print(f"\nQuery: [green]{query}[/green]")
    if response.query_expansion:
        console.print(f"Expanded: [dim]{', '.join(response.query_expansion)}[/dim]")
    console.print(f"Candidates: {response.total_candidates:,} jobs (after filters)")
    console.print(f"Search time: [cyan]{response.search_time_ms:.0f}ms[/cyan]")

    if response.degraded:
        console.print("[yellow]⚠ Running in degraded mode (keyword search only)[/yellow]")

    if response.cache_hit:
        console.print("[dim]Cache hit[/dim]")

    if not response.results:
        console.print("\n[yellow]No results found. Try broadening your search.[/yellow]")
        return

    # Results table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Score", justify="right", style="green", width=7)
    table.add_column("Title", max_width=40)
    table.add_column("Company", max_width=20)
    table.add_column("Salary", justify="right", width=15)

    for job in response.results:
        salary = ""
        if job.salary_min and job.salary_max:
            salary = f"${job.salary_min:,}-${job.salary_max:,}"
        elif job.salary_min:
            salary = f"${job.salary_min:,}+"

        table.add_row(
            f"{job.similarity_score:.3f}",
            job.title[:40],
            (job.company_name[:20] if job.company_name else "N/A"),
            salary,
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]Tip: Use --json for programmatic access[/dim]")


# API server command


@app.command(name="api-serve")
def serve_api(
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    db_path: str = typer.Option(
        "data/mcf_jobs.db", "--db", help="Path to SQLite database"
    ),
    index_dir: str = typer.Option(
        "data/embeddings", "--index-dir", help="Path to FAISS indexes"
    ),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of worker processes (production)"
    ),
    cors_origins: str = typer.Option(
        "http://localhost:3000,http://localhost:5173",
        "--cors",
        help="Comma-separated CORS origins",
    ),
    api_rate_limit: int = typer.Option(
        100,
        "--rate-limit",
        help="Max requests per minute per IP (0 to disable)",
    ),
) -> None:
    """
    Start the semantic search API server.

    Examples:
        mcf api-serve                     # Start on localhost:8000
        mcf api-serve --port 9000         # Custom port
        mcf api-serve --reload            # Auto-reload for dev
        mcf api-serve --workers 4         # Production mode
    """
    import os
    import uvicorn

    origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

    # Check prerequisites
    if not Path(db_path).exists():
        console.print(f"[red]Error:[/red] Database not found: {db_path}")
        console.print("Run 'mcf scrape' first to populate the database.")
        raise typer.Exit(1)

    index_path = Path(index_dir)
    if not (index_path / "jobs.index").exists():
        console.print(
            f"[yellow]Warning:[/yellow] FAISS index not found in {index_dir}"
        )
        console.print("API will run in degraded mode (keyword search only).")
        console.print("Run 'mcf embed-generate' to enable semantic search.")

    # Display startup info
    console.print("\n[bold green]Starting MCF Semantic Search API[/bold green]")
    console.print(f"  Database:   {db_path}")
    console.print(f"  Index dir:  {index_dir}")
    console.print(f"  Endpoint:   http://{host}:{port}")
    console.print(f"  API docs:   http://{host}:{port}/docs")
    console.print(f"  CORS:       {', '.join(origins)}")
    if api_rate_limit > 0:
        console.print(f"  Rate limit: {api_rate_limit} req/min per IP")
    else:
        console.print("  Rate limit: [yellow]disabled[/yellow]")
    if reload:
        console.print("  Mode:       [yellow]Development (auto-reload)[/yellow]")
    else:
        console.print(f"  Mode:       Production ({workers} worker{'s' if workers != 1 else ''})")
    console.print()

    # Pass config via environment so uvicorn workers can pick it up
    os.environ["MCF_DB_PATH"] = db_path
    os.environ["MCF_INDEX_DIR"] = index_dir
    os.environ["MCF_CORS_ORIGINS"] = cors_origins
    os.environ["MCF_RATE_LIMIT_RPM"] = str(api_rate_limit)

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        log_level="info",
    )


if __name__ == "__main__":
    app()
