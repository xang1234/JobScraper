import axios from 'axios';
import type {
  SearchRequest,
  SearchResponse,
  SimilarJobsRequest,
  SimilarBatchRequest,
  SimilarBatchResponse,
  SkillSearchRequest,
  CompanySimilarityRequest,
  CompanySimilarity,
  SkillCloudResponse,
  RelatedSkillsResponse,
  StatsResponse,
  HealthResponse,
  PopularQuery,
  PerformanceStats,
  ErrorResponse,
} from '@/types/api';

const client = axios.create({
  baseURL: '/',
  headers: { 'Content-Type': 'application/json' },
});

// Unwrap the API's { error: { code, message } } envelope on failures
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error) && error.response?.data?.error) {
      const apiError = error.response.data as ErrorResponse;
      return Promise.reject(new ApiError(
        apiError.error.message,
        apiError.error.code,
        error.response.status,
      ));
    }
    return Promise.reject(error);
  },
);

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(message: string, code: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

// =============================================================================
// Core Search
// =============================================================================

export async function searchJobs(req: SearchRequest): Promise<SearchResponse> {
  const { data } = await client.post<SearchResponse>('/api/search', req);
  return data;
}

export async function findSimilarJobs(req: SimilarJobsRequest): Promise<SearchResponse> {
  const { data } = await client.post<SearchResponse>('/api/similar', req);
  return data;
}

export async function findSimilarJobsBatch(req: SimilarBatchRequest): Promise<SimilarBatchResponse> {
  const { data } = await client.post<SimilarBatchResponse>('/api/similar/batch', req);
  return data;
}

export async function searchBySkill(req: SkillSearchRequest): Promise<SearchResponse> {
  const { data } = await client.post<SearchResponse>('/api/search/skills', req);
  return data;
}

// =============================================================================
// Skills
// =============================================================================

export async function getSkillCloud(minJobs = 10, limit = 100): Promise<SkillCloudResponse> {
  const { data } = await client.get<SkillCloudResponse>('/api/skills/cloud', {
    params: { min_jobs: minJobs, limit },
  });
  return data;
}

export async function getRelatedSkills(skill: string, k = 10): Promise<RelatedSkillsResponse> {
  const { data } = await client.get<RelatedSkillsResponse>(
    `/api/skills/related/${encodeURIComponent(skill)}`,
    { params: { k } },
  );
  return data;
}

// =============================================================================
// Companies
// =============================================================================

export async function findSimilarCompanies(req: CompanySimilarityRequest): Promise<CompanySimilarity[]> {
  const { data } = await client.post<CompanySimilarity[]>('/api/companies/similar', req);
  return data;
}

// =============================================================================
// Analytics & Utility
// =============================================================================

export async function getStats(): Promise<StatsResponse> {
  const { data } = await client.get<StatsResponse>('/api/stats');
  return data;
}

export async function getPopularQueries(days = 7, limit = 20): Promise<PopularQuery[]> {
  const { data } = await client.get<PopularQuery[]>('/api/analytics/popular', {
    params: { days, limit },
  });
  return data;
}

export async function getPerformanceStats(days = 7): Promise<PerformanceStats> {
  const { data } = await client.get<PerformanceStats>('/api/analytics/performance', {
    params: { days },
  });
  return data;
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>('/health');
  return data;
}
