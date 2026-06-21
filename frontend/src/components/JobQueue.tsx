/**
 * Job queue grid — lists all active jobs and provides batch controls.
 */
import { downloadAll } from '@/api/client'
import { useJobsQuery } from '@/hooks/useJobs'
import type { Job } from '@/types'
import { JobCard } from './JobCard'

function EmptyQueue() {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-studio-100">
        <svg className="h-8 w-8 text-studio-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
        </svg>
      </div>
      <h3 className="mt-4 text-sm font-semibold text-studio-800">No photos yet</h3>
      <p className="mt-1 text-sm text-studio-500">
        Drop car photos above and they'll appear here as they process.
      </p>
    </div>
  )
}

export function JobQueue() {
  const { data, isLoading, isError, error } = useJobsQuery()

  const jobs: Job[] = data?.jobs ?? []
  const completedJobs = jobs.filter((j) => j.status === 'completed')
  const hasCompleted = completedJobs.length > 0

  if (isLoading && jobs.length === 0) {
    return (
      <div className="flex justify-center py-12">
        <svg className="h-6 w-6 animate-spin text-studio-400" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-center">
        <p className="text-sm font-medium text-red-700">Failed to load jobs</p>
        <p className="mt-1 text-xs text-red-600">
          {error instanceof Error ? error.message : 'Unknown error'}
        </p>
      </div>
    )
  }

  if (jobs.length === 0) return <EmptyQueue />

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-studio-600">
          <span className="font-semibold text-studio-900">{jobs.length}</span> job{jobs.length !== 1 ? 's' : ''}
          {completedJobs.length > 0 && (
            <span className="ml-1.5 text-emerald-600">
              · {completedJobs.length} ready
            </span>
          )}
        </p>

        {hasCompleted && (
          <button
            type="button"
            onClick={() => downloadAll(completedJobs)}
            className="flex items-center gap-1.5 rounded-lg bg-studio-900 px-3.5 py-2 text-xs font-medium text-white transition-colors hover:bg-studio-800"
          >
            <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download all ({completedJobs.length})
          </button>
        )}
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {jobs.map((job) => (
          <JobCard key={job.id} job={job} />
        ))}
      </div>
    </div>
  )
}
