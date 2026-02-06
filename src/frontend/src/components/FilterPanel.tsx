import type { Filters } from '@/types/api'

const EMPLOYMENT_TYPES = ['', 'Full Time', 'Part Time', 'Contract', 'Temporary', 'Freelance']

interface FilterPanelProps {
  filters: Filters
  onChange: (filters: Filters) => void
}

export default function FilterPanel({ filters, onChange }: FilterPanelProps) {
  function update(patch: Partial<Filters>) {
    onChange({ ...filters, ...patch })
  }

  function handleClear() {
    onChange({ salary_min: null, salary_max: null, employment_type: null, company: null })
  }

  const hasFilters = filters.salary_min != null ||
    filters.salary_max != null ||
    filters.employment_type != null ||
    filters.company != null

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-900">Filters</h3>
        {hasFilters && (
          <button
            type="button"
            onClick={handleClear}
            className="text-xs text-blue-600 hover:text-blue-800"
          >
            Clear all
          </button>
        )}
      </div>

      {/* Salary Min */}
      <div>
        <label htmlFor="salary-min" className="block text-xs font-medium text-gray-700">
          Min Salary
        </label>
        <input
          id="salary-min"
          type="number"
          min={0}
          step={500}
          placeholder="e.g. 5000"
          value={filters.salary_min ?? ''}
          onChange={(e) =>
            update({ salary_min: e.target.value ? Number(e.target.value) : null })
          }
          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
        />
      </div>

      {/* Salary Max */}
      <div>
        <label htmlFor="salary-max" className="block text-xs font-medium text-gray-700">
          Max Salary
        </label>
        <input
          id="salary-max"
          type="number"
          min={0}
          step={500}
          placeholder="e.g. 15000"
          value={filters.salary_max ?? ''}
          onChange={(e) =>
            update({ salary_max: e.target.value ? Number(e.target.value) : null })
          }
          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
        />
      </div>

      {/* Employment Type */}
      <div>
        <label htmlFor="employment-type" className="block text-xs font-medium text-gray-700">
          Employment Type
        </label>
        <select
          id="employment-type"
          value={filters.employment_type ?? ''}
          onChange={(e) =>
            update({ employment_type: e.target.value || null })
          }
          className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-900 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
        >
          {EMPLOYMENT_TYPES.map((t) => (
            <option key={t} value={t}>
              {t || 'All types'}
            </option>
          ))}
        </select>
      </div>

      {/* Company */}
      <div>
        <label htmlFor="company-filter" className="block text-xs font-medium text-gray-700">
          Company
        </label>
        <input
          id="company-filter"
          type="text"
          placeholder="Filter by company"
          value={filters.company ?? ''}
          onChange={(e) =>
            update({ company: e.target.value || null })
          }
          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none"
        />
      </div>
    </div>
  )
}
