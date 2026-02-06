"""
FastAPI application for MCF semantic job search.

Wraps the synchronous SemanticSearchEngine with an async HTTP API.
The engine does CPU-bound work (FAISS, numpy, sqlite3), so search
handlers use run_in_executor to avoid blocking the event loop.

Usage:
    from src.api.app import create_app
    app = create_app(db_path="data/mcf_jobs.db", index_dir="data/embeddings")

    # Or run directly:
    # uvicorn src.api.app:app --reload
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .middleware import RateLimitMiddleware, RequestLoggingMiddleware
from .models import (
    CompanySimilarity,
    CompanySimilarityRequest,
    ErrorResponse,
    HealthResponse,
    JobResult,
    SearchRequest,
    SearchResponse,
    SimilarBatchRequest,
    SimilarBatchResponse,
    SimilarJobsRequest,
    SkillSearchRequest,
    StatsResponse,
)
from ..mcf.embeddings import SemanticSearchEngine
from ..mcf.embeddings.models import SimilarJobsRequest as InternalSimilarJobsRequest

logger = logging.getLogger(__name__)

# Global search engine instance (set during lifespan)
_search_engine: Optional[SemanticSearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load search indexes on startup, cleanup on shutdown."""
    global _search_engine

    db_path = app.state.db_path
    index_dir = Path(app.state.index_dir)

    logger.info("Loading search indexes (db=%s, index_dir=%s)...", db_path, index_dir)
    _search_engine = SemanticSearchEngine(db_path=db_path, index_dir=index_dir)

    # Load in a thread to avoid blocking the event loop during startup
    loop = asyncio.get_running_loop()
    loaded = await loop.run_in_executor(None, _search_engine.load)

    if loaded:
        logger.info("Search indexes loaded successfully")
    else:
        logger.warning("Search engine running in degraded mode (keyword-only)")

    yield

    logger.info("Shutting down search engine")
    _search_engine = None


def get_engine() -> SemanticSearchEngine:
    """FastAPI dependency — returns the loaded search engine."""
    if _search_engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    return _search_engine


def create_app(
    db_path: str | None = None,
    index_dir: str | None = None,
    cors_origins: Optional[list[str]] = None,
    rate_limit_rpm: int | None = None,
) -> FastAPI:
    """
    Application factory.

    Args:
        db_path: Path to the SQLite database.  Falls back to
            ``MCF_DB_PATH`` env var, then ``"data/mcf_jobs.db"``.
        index_dir: Directory containing FAISS indexes.  Falls back to
            ``MCF_INDEX_DIR`` env var, then ``"data/embeddings"``.
        cors_origins: Allowed CORS origins.  Falls back to
            ``MCF_CORS_ORIGINS`` env var (comma-separated), then
            common localhost ports.
        rate_limit_rpm: Max requests per minute per IP.  Falls back to
            ``MCF_RATE_LIMIT_RPM`` env var, then ``100``.  Set to ``0``
            to disable rate limiting.
    """
    import os

    if db_path is None:
        db_path = os.environ.get("MCF_DB_PATH", "data/mcf_jobs.db")
    if index_dir is None:
        index_dir = os.environ.get("MCF_INDEX_DIR", "data/embeddings")
    if cors_origins is None:
        env_origins = os.environ.get("MCF_CORS_ORIGINS")
        if env_origins:
            cors_origins = [o.strip() for o in env_origins.split(",") if o.strip()]
    if rate_limit_rpm is None:
        rate_limit_rpm = int(os.environ.get("MCF_RATE_LIMIT_RPM", "100"))

    app = FastAPI(
        title="MCF Semantic Search API",
        description="Semantic job search with hybrid BM25 + vector ranking",
        version="1.0.0",
        lifespan=lifespan,
        responses={
            400: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )

    # Store config on app state so lifespan can access it
    app.state.db_path = db_path
    app.state.index_dir = index_dir
    app.state.rate_limit_rpm = rate_limit_rpm

    if cors_origins is None:
        cors_origins = ["http://localhost:3000", "http://localhost:5173"]

    # Middleware execution order (outermost first):
    #   Logging → CORS → RateLimit → App
    # add_middleware prepends, so we add in reverse order.

    # 1. Rate limiting (innermost — added first)
    if rate_limit_rpm > 0:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=rate_limit_rpm)

    # 2. CORS (wraps rate limiter — 429s get CORS headers)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. Request logging (outermost — sees everything)
    app.add_middleware(RequestLoggingMiddleware)

    _register_routes(app)
    _register_exception_handlers(app)

    return app


# =========================================================================
# Route registration
# =========================================================================


def _register_routes(app: FastAPI) -> None:
    """Attach all route handlers to the app."""

    # -- Core search endpoints ------------------------------------------------

    @app.post("/api/search", response_model=SearchResponse)
    async def semantic_search(
        request: SearchRequest,
        engine: SemanticSearchEngine = Depends(get_engine),
    ) -> SearchResponse:
        """
        Semantic job search with filters.

        Combines SQL filtering, vector similarity, BM25 keyword ranking,
        query expansion, and freshness boosting.
        """
        internal_req = request.to_internal()
        loop = asyncio.get_running_loop()
        try:
            internal_resp = await loop.run_in_executor(None, engine.search, internal_req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return SearchResponse.from_internal(internal_resp)

    @app.post("/api/similar", response_model=SearchResponse)
    async def similar_jobs(
        request: SimilarJobsRequest,
        engine: SemanticSearchEngine = Depends(get_engine),
    ) -> SearchResponse:
        """Find jobs similar to a given job UUID."""
        internal_req = request.to_internal()
        loop = asyncio.get_running_loop()
        internal_resp = await loop.run_in_executor(
            None, engine.find_similar, internal_req
        )
        return SearchResponse.from_internal(internal_resp)

    @app.post("/api/similar/batch", response_model=SimilarBatchResponse)
    async def similar_jobs_batch(
        request: SimilarBatchRequest,
        engine: SemanticSearchEngine = Depends(get_engine),
    ) -> SimilarBatchResponse:
        """
        Find similar jobs for multiple UUIDs in one call.

        Limited to 50 UUIDs per request.  Useful for recommendation carousels.
        """
        loop = asyncio.get_running_loop()

        async def _find_one(uuid: str) -> tuple[str, list[JobResult], float]:
            internal_req = InternalSimilarJobsRequest(
                job_uuid=uuid,
                limit=request.limit_per_job,
                exclude_same_company=request.exclude_same_company,
            )
            resp = await loop.run_in_executor(
                None, engine.find_similar, internal_req
            )
            return (
                uuid,
                [JobResult.from_internal(r) for r in resp.results],
                resp.search_time_ms,
            )

        tasks = [_find_one(uuid) for uuid in request.job_uuids]
        completed = await asyncio.gather(*tasks)

        results = {uuid: jobs for uuid, jobs, _ in completed}
        total_time = sum(ms for _, _, ms in completed)

        return SimilarBatchResponse(results=results, search_time_ms=total_time)

    @app.post("/api/search/skills", response_model=SearchResponse)
    async def skill_search(
        request: SkillSearchRequest,
        engine: SemanticSearchEngine = Depends(get_engine),
    ) -> SearchResponse:
        """Search jobs by skill similarity."""
        internal_req = request.to_internal()
        loop = asyncio.get_running_loop()
        internal_resp = await loop.run_in_executor(
            None, engine.search_by_skill, internal_req
        )
        return SearchResponse.from_internal(internal_resp)

    @app.post("/api/companies/similar", response_model=list[CompanySimilarity])
    async def similar_companies(
        request: CompanySimilarityRequest,
        engine: SemanticSearchEngine = Depends(get_engine),
    ) -> list[CompanySimilarity]:
        """Find companies with similar job profiles."""
        internal_req = request.to_internal()
        loop = asyncio.get_running_loop()
        internal_results = await loop.run_in_executor(
            None, engine.find_similar_companies, internal_req
        )
        return [CompanySimilarity.from_internal(r) for r in internal_results]

    # -- Utility endpoints ----------------------------------------------------

    @app.get("/api/stats", response_model=StatsResponse)
    async def get_stats(
        engine: SemanticSearchEngine = Depends(get_engine),
    ) -> StatsResponse:
        """Get system statistics (index size, coverage, etc.)."""
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, engine.get_stats)
        emb = raw.get("embedding_stats", {})
        idx = raw.get("index_stats", {})
        return StatsResponse(
            total_jobs=emb.get("total_jobs", 0),
            jobs_with_embeddings=emb.get("job_embeddings", 0),
            embedding_coverage_pct=emb.get("coverage_pct", 0.0),
            unique_skills=emb.get("skill_embeddings", 0),
            unique_companies=emb.get("unique_companies", 0),
            index_size_mb=idx.get("index_size_mb", 0.0),
            model_version=raw.get("model_version", "unknown"),
        )

    @app.get("/api/analytics/popular")
    async def popular_queries(
        days: int = 7,
        limit: int = 20,
        engine: SemanticSearchEngine = Depends(get_engine),
    ) -> list[dict]:
        """Get most popular search queries."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(engine.db.get_popular_queries, days=days, limit=limit)
        )

    @app.get("/api/analytics/performance")
    async def performance_stats(
        days: int = 7,
        engine: SemanticSearchEngine = Depends(get_engine),
    ) -> dict:
        """Get search latency percentiles (p50, p90, p95, p99)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(engine.db.get_search_latency_percentiles, days=days)
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        if _search_engine is None:
            return HealthResponse(status="degraded", index_loaded=False, degraded=True)

        degraded = _search_engine._degraded
        return HealthResponse(
            status="healthy" if not degraded else "degraded",
            index_loaded=_search_engine._loaded,
            degraded=degraded,
        )


# =========================================================================
# Exception handlers
# =========================================================================

_STATUS_CODES = {
    400: "VALIDATION_ERROR",
    404: "NOT_FOUND",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


def _register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": _STATUS_CODES.get(exc.status_code, "UNKNOWN_ERROR"),
                    "message": exc.detail,
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                }
            },
        )


# =========================================================================
# Default app instance (for `uvicorn src.api.app:app`)
# =========================================================================

app = create_app()
