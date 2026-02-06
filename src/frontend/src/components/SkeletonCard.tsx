export default function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-lg border border-gray-200 bg-white p-5">
      {/* Title */}
      <div className="h-5 w-3/4 rounded bg-gray-200" />
      {/* Company */}
      <div className="mt-2 h-4 w-1/2 rounded bg-gray-200" />
      {/* Salary + Location row */}
      <div className="mt-3 flex gap-4">
        <div className="h-4 w-24 rounded bg-gray-200" />
        <div className="h-4 w-32 rounded bg-gray-200" />
      </div>
      {/* Skill tags */}
      <div className="mt-3 flex gap-2">
        <div className="h-6 w-16 rounded-full bg-gray-200" />
        <div className="h-6 w-20 rounded-full bg-gray-200" />
        <div className="h-6 w-14 rounded-full bg-gray-200" />
      </div>
    </div>
  )
}
