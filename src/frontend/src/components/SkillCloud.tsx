import type { SkillCloudItem } from '@/types/api'

// 8 visually distinct colors mapped by cluster_id
const CLUSTER_COLORS = [
  'text-blue-600',
  'text-emerald-600',
  'text-purple-600',
  'text-orange-600',
  'text-rose-600',
  'text-teal-600',
  'text-amber-600',
  'text-indigo-600',
]

interface SkillCloudProps {
  items: SkillCloudItem[]
  onSkillClick: (skill: string) => void
}

export default function SkillCloud({ items, onSkillClick }: SkillCloudProps) {
  if (items.length === 0) return null

  const maxCount = Math.max(...items.map((i) => i.job_count))
  const minCount = Math.min(...items.map((i) => i.job_count))
  const range = maxCount - minCount || 1

  function fontSize(count: number): string {
    // Scale between 0.75rem and 1.5rem based on job count
    const ratio = (count - minCount) / range
    const size = 0.75 + ratio * 0.75
    return `${size}rem`
  }

  function colorClass(clusterId: number | null): string {
    if (clusterId == null) return 'text-gray-600'
    return CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length]
  }

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-900 mb-3">Skills</h3>
      <div className="flex flex-wrap gap-x-2 gap-y-1">
        {items.map((item) => (
          <button
            key={item.skill}
            type="button"
            onClick={() => onSkillClick(item.skill)}
            style={{ fontSize: fontSize(item.job_count) }}
            className={`hover:underline cursor-pointer font-medium leading-relaxed ${colorClass(item.cluster_id)}`}
            title={`${item.job_count} jobs`}
          >
            {item.skill}
          </button>
        ))}
      </div>
    </div>
  )
}
