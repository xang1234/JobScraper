// =============================================================================
// Request Types
// =============================================================================

export interface SearchRequest {
  query: string;
  limit?: number;
  salary_min?: number | null;
  salary_max?: number | null;
  employment_type?: string | null;
  region?: string | null;
  company?: string | null;
  alpha?: number;
  freshness_weight?: number;
  expand_query?: boolean;
  min_similarity?: number;
}

export interface SimilarJobsRequest {
  job_uuid: string;
  limit?: number;
  exclude_same_company?: boolean;
  freshness_weight?: number;
}

export interface SimilarBatchRequest {
  job_uuids: string[];
  limit_per_job?: number;
  exclude_same_company?: boolean;
}

export interface SkillSearchRequest {
  skill: string;
  limit?: number;
  salary_min?: number | null;
  salary_max?: number | null;
  employment_type?: string | null;
}

export interface CompanySimilarityRequest {
  company_name: string;
  limit?: number;
}

// =============================================================================
// Response Types
// =============================================================================

export interface JobResult {
  uuid: string;
  title: string;
  company_name: string | null;
  description: string;
  salary_min: number | null;
  salary_max: number | null;
  employment_type: string | null;
  skills: string | null;
  location: string | null;
  posted_date: string | null;
  job_url: string | null;
  similarity_score: number;
  bm25_score: number | null;
  freshness_score: number | null;
}

export interface SearchResponse {
  results: JobResult[];
  total_candidates: number;
  search_time_ms: number;
  query_expansion: string[] | null;
  degraded: boolean;
  cache_hit: boolean;
}

export interface SimilarBatchResponse {
  results: Record<string, JobResult[]>;
  search_time_ms: number;
}

export interface SkillCloudItem {
  skill: string;
  job_count: number;
  cluster_id: number | null;
}

export interface SkillCloudResponse {
  items: SkillCloudItem[];
  total_unique_skills: number;
}

export interface RelatedSkill {
  skill: string;
  similarity: number;
  same_cluster: boolean;
}

export interface RelatedSkillsResponse {
  skill: string;
  related: RelatedSkill[];
}

export interface CompanySimilarity {
  company_name: string;
  similarity_score: number;
  job_count: number;
  avg_salary: number | null;
  top_skills: string[];
}

export interface StatsResponse {
  total_jobs: number;
  jobs_with_embeddings: number;
  embedding_coverage_pct: number;
  unique_skills: number;
  unique_companies: number;
  index_size_mb: number;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  index_loaded: boolean;
  degraded: boolean;
}

export interface PopularQuery {
  query: string;
  count: number;
}

export interface PerformanceStats {
  p50_ms: number;
  p90_ms: number;
  p95_ms: number;
  p99_ms: number;
  total_queries: number;
}

// =============================================================================
// UI Types
// =============================================================================

export interface Filters {
  salary_min: number | null;
  salary_max: number | null;
  employment_type: string | null;
  company: string | null;
}

// =============================================================================
// Error Types
// =============================================================================

export interface ErrorDetail {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface ErrorResponse {
  error: ErrorDetail;
}
