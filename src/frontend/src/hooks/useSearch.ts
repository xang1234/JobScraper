import { useState, useCallback } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import type { Filters, CompanySimilarity, JobResult } from '@/types/api'
import {
  searchJobs,
  findSimilarJobs,
  findSimilarCompanies,
  getSkillCloud,
  getHealth,
} from '@/services/api'

const emptyFilters: Filters = {
  salary_min: null,
  salary_max: null,
  employment_type: null,
  company: null,
}

export function useSearch() {
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<Filters>(emptyFilters)

  // ── Core search (cached by query + filters) ──
  const searchResult = useQuery({
    queryKey: ['search', query, filters],
    queryFn: () =>
      searchJobs({
        query,
        limit: 20,
        salary_min: filters.salary_min,
        salary_max: filters.salary_max,
        employment_type: filters.employment_type,
        company: filters.company,
        expand_query: true,
      }),
    enabled: !!query,
  })

  // ── Skill cloud (loads once on mount) ──
  const skillCloud = useQuery({
    queryKey: ['skillCloud'],
    queryFn: () => getSkillCloud(10, 80),
    staleTime: 10 * 60 * 1000, // 10 minutes
  })

  // ── Health check ──
  const health = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000, // poll every minute
  })

  // ── Similar jobs (one-shot imperative) ──
  const [similarJobs, setSimilarJobs] = useState<JobResult[] | undefined>()
  const [similarModalOpen, setSimilarModalOpen] = useState(false)

  const similarMutation = useMutation({
    mutationFn: (uuid: string) =>
      findSimilarJobs({ job_uuid: uuid, limit: 10, exclude_same_company: true }),
    onSuccess: (data) => {
      setSimilarJobs(data.results)
      setSimilarModalOpen(true)
    },
  })

  const handleFindSimilar = useCallback(
    (uuid: string) => {
      setSimilarJobs(undefined)
      setSimilarModalOpen(true)
      similarMutation.mutate(uuid)
    },
    [similarMutation],
  )

  const closeSimilarModal = useCallback(() => {
    setSimilarModalOpen(false)
  }, [])

  // ── Company search (one-shot imperative) ──
  const [companyResults, setCompanyResults] = useState<CompanySimilarity[] | undefined>()

  const companyMutation = useMutation({
    mutationFn: (company: string) =>
      findSimilarCompanies({ company_name: company, limit: 10 }),
    onSuccess: (data) => {
      setCompanyResults(data)
    },
  })

  const handleCompanySearch = useCallback(
    (company: string) => {
      setCompanyResults(undefined)
      companyMutation.mutate(company)
    },
    [companyMutation],
  )

  // ── Search trigger ──
  const handleSearch = useCallback((q: string) => {
    setQuery(q)
  }, [])

  const handleSkillClick = useCallback((skill: string) => {
    setQuery(skill)
  }, [])

  return {
    // Search
    query,
    filters,
    setFilters,
    handleSearch,
    searchResult,
    hasSearched: !!query,

    // Skill cloud
    skillCloud,
    handleSkillClick,

    // Health
    health,

    // Similar jobs modal
    similarModalOpen,
    similarJobs,
    similarLoading: similarMutation.isPending,
    handleFindSimilar,
    closeSimilarModal,

    // Company search
    companyResults,
    companyLoading: companyMutation.isPending,
    handleCompanySearch,
  }
}
