"""
Performance benchmarking for the semantic search system.

Measures three key areas:
- Search latency (p95 target: <100ms, cached p95: <20ms)
- Query embedding time (target: <50ms per single query)
- FAISS index load time (target: <5s)

Usage:
    python scripts/benchmark.py --queries 100 --warmup 10
    python scripts/benchmark.py --queries 50 --db data/mcf_jobs.db
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Add project root to path so we can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mcf.embeddings import FAISSIndexManager, SemanticSearchEngine
from src.mcf.embeddings.generator import EmbeddingGenerator
from src.mcf.embeddings.models import SearchRequest

console = Console()

# Diverse queries covering different job domains
BENCHMARK_QUERIES = [
    "python developer",
    "machine learning engineer",
    "data scientist singapore",
    "full stack javascript react",
    "devops kubernetes aws",
    "backend java spring boot",
    "frontend react typescript",
    "data analyst sql tableau",
    "product manager agile",
    "cloud architect azure",
]


def benchmark_search(
    engine: SemanticSearchEngine,
    n_queries: int,
    warmup: int,
) -> dict:
    """Benchmark search latency with warmup phase."""
    queries = [
        BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
        for i in range(n_queries + warmup)
    ]

    # Warmup: populate caches and JIT-compile hot paths
    console.print(f"[dim]Warming up with {warmup} queries...[/dim]")
    for q in queries[:warmup]:
        engine.search(SearchRequest(query=q, limit=10))

    # Benchmark fresh queries (no cache)
    console.print(f"[dim]Benchmarking {n_queries} fresh queries...[/dim]")
    engine.clear_caches()
    fresh_latencies = []

    for q in queries[warmup:]:
        start = time.perf_counter()
        engine.search(SearchRequest(query=q, limit=10))
        elapsed_ms = (time.perf_counter() - start) * 1000
        fresh_latencies.append(elapsed_ms)

    # Benchmark cached queries (repeat same queries)
    console.print(f"[dim]Benchmarking {n_queries} cached queries...[/dim]")
    cached_latencies = []

    for q in queries[warmup:]:
        start = time.perf_counter()
        response = engine.search(SearchRequest(query=q, limit=10))
        elapsed_ms = (time.perf_counter() - start) * 1000
        if response.cache_hit:
            cached_latencies.append(elapsed_ms)

    fresh_latencies.sort()
    cached_latencies.sort()

    def percentiles(latencies: list[float]) -> dict:
        n = len(latencies)
        if n == 0:
            return {}
        return {
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p90": latencies[int(n * 0.9)],
            "p95": latencies[int(n * 0.95)],
            "p99": latencies[int(n * 0.99)] if n >= 100 else latencies[-1],
        }

    return {
        "fresh": percentiles(fresh_latencies),
        "cached": percentiles(cached_latencies),
        "cache_hit_rate": len(cached_latencies) / n_queries * 100,
    }


def benchmark_embedding(generator: EmbeddingGenerator, n_texts: int) -> dict:
    """Benchmark single and batch embedding generation time."""
    texts = [
        f"We are looking for a {BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]} "
        f"to join our team. Requirements include Python, SQL, and machine learning."
        for i in range(n_texts)
    ]

    # Warmup: ensure model is loaded before timing
    console.print("[dim]Warming up embedding model...[/dim]")
    generator.model.encode("warmup query")

    # Single embedding latency (average of 10 runs)
    single_times = []
    for text in texts[:10]:
        start = time.perf_counter()
        generator.model.encode(text)
        single_times.append((time.perf_counter() - start) * 1000)

    # Batch embedding throughput
    start = time.perf_counter()
    generator.model.encode(texts, batch_size=32)
    batch_total_ms = (time.perf_counter() - start) * 1000

    return {
        "single_ms": statistics.mean(single_times),
        "single_p95_ms": sorted(single_times)[int(len(single_times) * 0.95)],
        "batch_per_item_ms": batch_total_ms / n_texts,
        "batch_total_ms": batch_total_ms,
        "batch_size": n_texts,
    }


def benchmark_index_load(index_dir: Path) -> dict:
    """Benchmark FAISS index load time from disk."""
    manager = FAISSIndexManager(index_dir)

    if not manager.exists():
        return {"error": f"No indexes found at {index_dir}"}

    start = time.perf_counter()
    manager.load()
    load_time = time.perf_counter() - start

    stats = manager.get_stats()
    job_stats = stats.get("indexes", {}).get("jobs", {})

    return {
        "load_time_s": load_time,
        "index_size_mb": job_stats.get("estimated_memory_mb", 0),
        "n_vectors": job_stats.get("total_vectors", 0),
        "index_type": job_stats.get("index_type", "unknown"),
    }


def status_icon(value: float, target: float, lower_is_better: bool = True) -> str:
    """Return pass/fail icon based on target comparison."""
    if lower_is_better:
        return "[green]PASS[/green]" if value <= target else "[red]FAIL[/red]"
    return "[green]PASS[/green]" if value >= target else "[red]FAIL[/red]"


def print_results(
    search_results: dict,
    embedding_results: dict,
    index_results: dict,
) -> bool:
    """Print benchmark results and return True if all targets met."""
    all_passed = True

    # --- Search latency (fresh) ---
    table = Table(title="Search Latency — Fresh (ms)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status", justify="center")

    fresh = search_results["fresh"]
    if fresh:
        for key in ("min", "mean", "median", "p90"):
            table.add_row(key.upper(), f"{fresh[key]:.1f}", "-", "")

        p95_pass = fresh["p95"] <= 100
        all_passed &= p95_pass
        table.add_row(
            "P95", f"{fresh['p95']:.1f}", "<100", status_icon(fresh["p95"], 100)
        )

        if "p99" in fresh:
            table.add_row("P99", f"{fresh['p99']:.1f}", "-", "")
        table.add_row("MAX", f"{fresh['max']:.1f}", "-", "")
    else:
        table.add_row("(no data)", "-", "-", "")

    console.print(table)
    console.print()

    # --- Search latency (cached) ---
    table = Table(title="Search Latency — Cached (ms)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status", justify="center")

    cached = search_results["cached"]
    if cached:
        for key in ("min", "mean", "median"):
            table.add_row(key.upper(), f"{cached[key]:.1f}", "-", "")

        p95_pass = cached["p95"] <= 20
        all_passed &= p95_pass
        table.add_row(
            "P95", f"{cached['p95']:.1f}", "<20", status_icon(cached["p95"], 20)
        )
        table.add_row("MAX", f"{cached['max']:.1f}", "-", "")
    else:
        table.add_row("(no cache hits)", "-", "-", "")

    cache_rate = search_results["cache_hit_rate"]
    cache_pass = cache_rate >= 30
    all_passed &= cache_pass
    table.add_row(
        "Cache Hit Rate",
        f"{cache_rate:.1f}%",
        ">30%",
        status_icon(cache_rate, 30, lower_is_better=False),
    )

    console.print(table)
    console.print()

    # --- Embedding generation ---
    table = Table(title="Embedding Generation (ms)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status", justify="center")

    single_pass = embedding_results["single_ms"] <= 50
    all_passed &= single_pass
    table.add_row(
        "Single (mean)",
        f"{embedding_results['single_ms']:.1f}",
        "<50",
        status_icon(embedding_results["single_ms"], 50),
    )
    table.add_row(
        "Batch (per item)",
        f"{embedding_results['batch_per_item_ms']:.1f}",
        "-",
        "",
    )
    table.add_row(
        f"Batch total ({embedding_results['batch_size']} items)",
        f"{embedding_results['batch_total_ms']:.0f}",
        "-",
        "",
    )

    console.print(table)
    console.print()

    # --- Index loading ---
    table = Table(title="Index Loading")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status", justify="center")

    if "error" in index_results:
        table.add_row("Error", index_results["error"], "-", "[yellow]SKIP[/yellow]")
    else:
        load_pass = index_results["load_time_s"] <= 5
        all_passed &= load_pass
        table.add_row(
            "Load Time",
            f"{index_results['load_time_s']:.2f}s",
            "<5s",
            status_icon(index_results["load_time_s"], 5),
        )
        table.add_row(
            "Memory (est.)",
            f"{index_results['index_size_mb']:.1f} MB",
            "-",
            "",
        )
        table.add_row(
            "Vectors",
            f"{index_results['n_vectors']:,}",
            "-",
            "",
        )
        table.add_row("Index Type", index_results["index_type"], "-", "")

    console.print(table)
    console.print()

    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark semantic search performance"
    )
    parser.add_argument(
        "--queries", type=int, default=100, help="Number of benchmark queries"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup queries"
    )
    parser.add_argument(
        "--embed-texts",
        type=int,
        default=100,
        help="Number of texts for embedding benchmark",
    )
    parser.add_argument(
        "--db", type=str, default="data/mcf_jobs.db", help="Database path"
    )
    parser.add_argument(
        "--index-dir", type=str, default="data/embeddings", help="Index directory"
    )
    args = parser.parse_args()

    console.print("[bold]MCF Semantic Search Benchmark[/bold]")
    console.print()

    # --- Load engine ---
    console.print("[dim]Loading search engine...[/dim]")
    index_dir = Path(args.index_dir)
    engine = SemanticSearchEngine(args.db, index_dir)
    engine.load()

    if engine._degraded:
        console.print(
            "[yellow]Warning: Engine running in degraded mode "
            "(keyword-only, no vector search). "
            "Latency numbers will not reflect full hybrid search.[/yellow]\n"
        )

    # --- Run benchmarks ---
    search_results = benchmark_search(engine, args.queries, args.warmup)
    embedding_results = benchmark_embedding(engine.generator, args.embed_texts)
    index_results = benchmark_index_load(index_dir)

    # --- Print results ---
    all_passed = print_results(search_results, embedding_results, index_results)

    if all_passed:
        console.print("[bold green]All performance targets met![/bold green]")
        return 0
    else:
        console.print("[bold red]Some performance targets not met[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
