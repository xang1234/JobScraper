import { ExclamationTriangleIcon } from '@heroicons/react/24/outline'

export default function DegradedBanner({ show }: { show: boolean }) {
  if (!show) return null

  return (
    <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
      <div className="flex items-center">
        <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400 shrink-0" />
        <p className="ml-3 text-sm text-yellow-700">
          Search is running in degraded mode. Some features may be unavailable or
          results may be less accurate.
        </p>
      </div>
    </div>
  )
}
