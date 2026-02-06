import type { JobResult } from '@/types/api'
import {
  MapPinIcon,
  BriefcaseIcon,
  CurrencyDollarIcon,
} from '@heroicons/react/24/outline'

interface JobCardProps {
  job: JobResult
  onFindSimilar: (uuid: string) => void
}

function formatSalary(min: number | null, max: number | null): string | null {
  if (min != null && max != null) {
    return `$${min.toLocaleString()} – $${max.toLocaleString()}`
  }
  if (min != null) return `From $${min.toLocaleString()}`
  if (max != null) return `Up to $${max.toLocaleString()}`
  return null
}

function similarityColor(score: number): string {
  if (score >= 0.8) return 'bg-green-50 text-green-700 ring-green-600/20'
  if (score >= 0.6) return 'bg-blue-50 text-blue-700 ring-blue-600/20'
  return 'bg-gray-50 text-gray-600 ring-gray-500/20'
}

const MAX_SKILLS = 5

export default function JobCard({ job, onFindSimilar }: JobCardProps) {
  const salary = formatSalary(job.salary_min, job.salary_max)
  const skills = job.skills
    ? job.skills.split(',').map((s) => s.trim()).filter(Boolean)
    : []
  const visibleSkills = skills.slice(0, MAX_SKILLS)
  const extraCount = skills.length - MAX_SKILLS

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 hover:shadow-md transition-shadow">
      {/* Header: title + similarity badge */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          {job.job_url ? (
            <a
              href={job.job_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-base font-semibold text-blue-600 hover:underline truncate block"
            >
              {job.title}
            </a>
          ) : (
            <span className="text-base font-semibold text-gray-900 truncate block">
              {job.title}
            </span>
          )}
          {job.company_name && (
            <p className="mt-0.5 text-sm text-gray-600">{job.company_name}</p>
          )}
        </div>
        <span
          className={`inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${similarityColor(job.similarity_score)}`}
        >
          {(job.similarity_score * 100).toFixed(0)}%
        </span>
      </div>

      {/* Meta row */}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm text-gray-500">
        {salary && (
          <span className="inline-flex items-center gap-1">
            <CurrencyDollarIcon className="h-4 w-4" />
            {salary}
          </span>
        )}
        {job.location && (
          <span className="inline-flex items-center gap-1">
            <MapPinIcon className="h-4 w-4" />
            {job.location}
          </span>
        )}
        {job.employment_type && (
          <span className="inline-flex items-center gap-1">
            <BriefcaseIcon className="h-4 w-4" />
            {job.employment_type}
          </span>
        )}
      </div>

      {/* Skills */}
      {visibleSkills.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {visibleSkills.map((skill) => (
            <span
              key={skill}
              className="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-700"
            >
              {skill}
            </span>
          ))}
          {extraCount > 0 && (
            <span className="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs text-gray-500">
              +{extraCount} more
            </span>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="mt-3">
        <button
          type="button"
          onClick={() => onFindSimilar(job.uuid)}
          className="text-sm font-medium text-blue-600 hover:text-blue-800"
        >
          Find Similar
        </button>
      </div>
    </div>
  )
}
