import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import type { JobResult } from '@/types/api'
import JobCard from '@/components/JobCard'
import SkeletonCard from '@/components/SkeletonCard'

interface SimilarJobsModalProps {
  open: boolean
  onClose: () => void
  jobs: JobResult[] | undefined
  isLoading: boolean
  onFindSimilar: (uuid: string) => void
}

export default function SimilarJobsModal({
  open,
  onClose,
  jobs,
  isLoading,
  onFindSimilar,
}: SimilarJobsModalProps) {
  return (
    <Dialog open={open} onClose={onClose} className="relative z-50">
      <DialogBackdrop className="fixed inset-0 bg-black/30 transition-opacity" />

      <div className="fixed inset-0 overflow-y-auto">
        <div className="flex min-h-full items-start justify-center p-4 pt-16">
          <DialogPanel className="w-full max-w-2xl rounded-xl bg-white p-6 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <DialogTitle className="text-lg font-semibold text-gray-900">
                Similar Jobs
              </DialogTitle>
              <button
                type="button"
                onClick={onClose}
                className="rounded-md p-1 text-gray-400 hover:text-gray-600"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>

            <div className="space-y-4">
              {isLoading ? (
                <>
                  <SkeletonCard />
                  <SkeletonCard />
                </>
              ) : jobs && jobs.length > 0 ? (
                jobs.map((job) => (
                  <JobCard key={job.uuid} job={job} onFindSimilar={onFindSimilar} />
                ))
              ) : (
                <p className="py-8 text-center text-gray-500">
                  No similar jobs found.
                </p>
              )}
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  )
}
