/**
 * React Query hooks for job data.
 *
 * Polling strategy:
 *   - Polls GET /api/jobs?ids=... every 3 s while any job is non-terminal.
 *   - Stops polling once all jobs reach completed/failed.
 *   - WebSocket updates (via useWebSocket) invalidate the query cache
 *     immediately for a faster perceived response.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { cancelJob, createJobs, getJobs } from '@/api/client'
import type { Job, JobListResponse } from '@/types'
import { useJobStore } from '@/store/jobStore'

const TERMINAL = new Set(['completed', 'failed'])
const POLL_INTERVAL_MS = 3_000

export const jobsQueryKey = (ids: string[]) => ['jobs', ids.sort().join(',')]

// ── Read: poll all tracked job IDs ────────────────────────────────────────────

export function useJobsQuery() {
  const jobIds = useJobStore((s) => s.jobIds)

  return useQuery<JobListResponse>({
    queryKey: jobsQueryKey(jobIds),
    queryFn: () => getJobs(jobIds),
    enabled: jobIds.length > 0,
    refetchInterval: (query) => {
      const jobs = query.state.data?.jobs ?? []
      const allDone = jobs.length > 0 && jobs.every((j) => TERMINAL.has(j.status))
      return allDone ? false : POLL_INTERVAL_MS
    },
    staleTime: 1_000,
  })
}

// ── Write: upload files → create jobs ────────────────────────────────────────

export function useCreateJobs() {
  const qc = useQueryClient()
  const addJobs = useJobStore((s) => s.addJobs)

  return useMutation({
    mutationFn: (files: File[]) => createJobs(files),
    onSuccess: (data) => {
      addJobs(data.jobs)
      qc.setQueryData<JobListResponse>(
        jobsQueryKey(data.jobs.map((j) => j.id)),
        data,
      )
    },
  })
}

// ── Write: cancel a pending job ───────────────────────────────────────────────

export function useCancelJob() {
  const qc = useQueryClient()
  const jobIds = useJobStore((s) => s.jobIds)

  return useMutation({
    mutationFn: (jobId: string) => cancelJob(jobId),
    onSuccess: (_data, jobId) => {
      // Optimistic update: mark as failed locally before refetch
      qc.setQueryData<JobListResponse>(jobsQueryKey(jobIds), (old) => {
        if (!old) return old
        return {
          ...old,
          jobs: old.jobs.map((j): Job =>
            j.id === jobId ? { ...j, status: 'failed', errorMessage: 'Cancelled by user' } : j,
          ),
        }
      })
    },
  })
}

// ── Helper: merge a WebSocket progress event into the query cache ─────────────

export function useApplyWebSocketUpdate() {
  const qc = useQueryClient()
  const jobIds = useJobStore((s) => s.jobIds)

  return (event: { jobId: string; status: string; progressPercent?: number; resultImageUrl?: string; errorMessage?: string }) => {
    qc.setQueryData<JobListResponse>(jobsQueryKey(jobIds), (old) => {
      if (!old) return old
      return {
        ...old,
        jobs: old.jobs.map((j): Job =>
          j.id === event.jobId
            ? {
                ...j,
                status: event.status as Job['status'],
                progressPercent: event.progressPercent ?? j.progressPercent,
                resultImageUrl: event.resultImageUrl ?? j.resultImageUrl,
                errorMessage: event.errorMessage ?? j.errorMessage,
                updatedAt: new Date().toISOString(),
              }
            : j,
        ),
      }
    })
  }
}
