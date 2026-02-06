import { useState } from 'react'
import { BuildingOfficeIcon } from '@heroicons/react/24/outline'
import type { CompanySimilarity } from '@/types/api'

interface CompanySearchProps {
  onSearch: (company: string) => void
  results: CompanySimilarity[] | undefined
  isLoading: boolean
}

export default function CompanySearch({ onSearch, results, isLoading }: CompanySearchProps) {
  const [input, setInput] = useState('')

  function handleSubmit() {
    const trimmed = input.trim()
    if (trimmed) onSearch(trimmed)
  }

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-900 mb-3">Company Search</h3>

      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleSubmit() }}
          placeholder="Company name"
          className="block w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
        />
        <button
          type="button"
          onClick={handleSubmit}
          disabled={isLoading || !input.trim()}
          className="rounded-md bg-gray-100 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Go
        </button>
      </div>

      {isLoading && (
        <div className="mt-3 text-sm text-gray-400">Searching...</div>
      )}

      {results && results.length > 0 && (
        <ul className="mt-3 space-y-2">
          {results.map((c) => (
            <li
              key={c.company_name}
              className="rounded-md border border-gray-200 p-3 text-sm"
            >
              <div className="flex items-center gap-2">
                <BuildingOfficeIcon className="h-4 w-4 text-gray-400 shrink-0" />
                <span className="font-medium text-gray-900 truncate">{c.company_name}</span>
                <span className="ml-auto shrink-0 text-xs text-gray-500">
                  {(c.similarity_score * 100).toFixed(0)}%
                </span>
              </div>
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-gray-500">
                <span>{c.job_count} jobs</span>
                {c.avg_salary != null && (
                  <span>avg ${c.avg_salary.toLocaleString()}</span>
                )}
              </div>
              {c.top_skills.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {c.top_skills.slice(0, 5).map((skill) => (
                    <span
                      key={skill}
                      className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}

      {results && results.length === 0 && (
        <p className="mt-3 text-sm text-gray-500">No similar companies found.</p>
      )}
    </div>
  )
}
