import { Header } from '@/components/Header'
import { UploadZone } from '@/components/UploadZone'
import { JobQueue } from '@/components/JobQueue'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useJobsQuery } from '@/hooks/useJobs'
import { useJobStore } from '@/store/jobStore'

export default function App() {
  // Connect WebSocket — auto-subscribes to job IDs as they're added to the store
  useWebSocket()

  const { data } = useJobsQuery()
  const jobIds = useJobStore((s) => s.jobIds)

  const jobs = data?.jobs ?? []
  const completedCount = jobs.filter((j) => j.status === 'completed').length
  const totalCount = jobs.length

  return (
    <div className="flex min-h-screen flex-col">
      <Header completedCount={completedCount} totalCount={totalCount} />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6 lg:px-8">
        {/* Page title */}
        <div className="mb-8">
          <h1 className="text-xl font-semibold text-studio-900">Studio Background Generator</h1>
          <p className="mt-1 text-sm text-studio-500">
            Upload car photos and receive them composited onto a professional dealership studio backdrop.
            The original vehicle is preserved pixel-perfectly — only the background changes.
          </p>
        </div>

        {/* Upload zone */}
        <section className="mb-8">
          <UploadZone />
        </section>

        {/* Job queue */}
        {jobIds.length > 0 && (
          <section>
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-studio-500">
              Processing Queue
            </h2>
            <JobQueue />
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-studio-200 py-4">
        <div className="mx-auto max-w-7xl px-4 text-center text-xs text-studio-400 sm:px-6 lg:px-8">
          Car Studio · AI Background Generator · Original vehicle pixels are never modified
        </div>
      </footer>
    </div>
  )
}
