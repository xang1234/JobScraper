import type { SearchResponse } from '@/types/api'

export default function SearchStats({ data }: { data: SearchResponse }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-gray-500">
      <span>
        <span className="font-medium text-gray-900">{data.total_candidates.toLocaleString()}</span>{' '}
        candidates
      </span>
      <span>{data.search_time_ms.toFixed(0)}ms</span>
      {data.cache_hit && (
        <span className="inline-flex items-center rounded-full bg-green-50 px-2 py-0.5 text-xs font-medium text-green-700 ring-1 ring-green-600/20 ring-inset">
          cached
        </span>
      )}
      {data.query_expansion && data.query_expansion.length > 0 && (
        <span className="text-gray-400">
          expanded: {data.query_expansion.join(', ')}
        </span>
      )}
    </div>
  )
}
