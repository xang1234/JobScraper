import type { JobResult } from '@/types/api'
import JobCard from '@/components/JobCard'
import SkeletonCard from '@/components/SkeletonCard'

interface JobListProps {
  jobs: JobResult[] | undefined
  isLoading: boolean
  hasSearched: boolean
  onFindSimilar: (uuid: string) => void
}

export default function JobList({ jobs, isLoading, hasSearched, onFindSimilar }: JobListProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <SkeletonCard />
        <SkeletonCard />
        <SkeletonCard />
      </div>
    )
  }

  if (!hasSearched) {
    return (
      <div className="py-12 text-center text-gray-400">
        Enter a search query to find jobs
      </div>
    )
  }

  if (!jobs || jobs.length === 0) {
    return (
      <div className="py-12 text-center text-gray-500">
        No jobs found. Try a different search or adjust your filters.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {jobs.map((job) => (
        <JobCard key={job.uuid} job={job} onFindSimilar={onFindSimilar} />
      ))}
    </div>
  )
}
