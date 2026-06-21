/**
 * Typed API client.
 *
 * All functions throw a descriptive Error on non-2xx responses so
 * React Query can surface the error message directly in the UI.
 */
import type { Job, JobListResponse } from '@/types'

const BASE = '/api'

// ── Helpers ────────────────────────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (res.ok) return res.json() as Promise<T>

  let detail = `HTTP ${res.status}`
  try {
    const body = await res.json()
    detail = body.detail ?? body.message ?? detail
  } catch {
    // ignore parse error — use status code message
  }
  throw new Error(detail)
}

// ── Job endpoints ──────────────────────────────────────────────────────────────

/**
 * Upload one or more files.  Returns immediately — jobs are processed async.
 */
export async function createJobs(files: File[]): Promise<JobListResponse> {
  const form = new FormData()
  for (const f of files) form.append('files', f)

  const res = await fetch(`${BASE}/jobs`, { method: 'POST', body: form })
  return handleResponse<JobListResponse>(res)
}

/**
 * Fetch a single job by ID.
 */
export async function getJob(id: string): Promise<Job> {
  const res = await fetch(`${BASE}/jobs/${id}`)
  return handleResponse<Job>(res)
}

/**
 * Batch-fetch multiple jobs.
 */
export async function getJobs(ids: string[]): Promise<JobListResponse> {
  if (ids.length === 0) return { jobs: [], total: 0 }
  const params = new URLSearchParams({ ids: ids.join(',') })
  const res = await fetch(`${BASE}/jobs?${params}`)
  return handleResponse<JobListResponse>(res)
}

/**
 * Cancel a pending job.
 */
export async function cancelJob(id: string): Promise<void> {
  const res = await fetch(`${BASE}/jobs/${id}`, { method: 'DELETE' })
  await handleResponse<unknown>(res)
}

/**
 * Trigger a browser download for the result image.
 */
export function downloadResult(job: Job): void {
  if (!job.resultImageUrl) return
  const a = document.createElement('a')
  a.href = job.resultImageUrl
  a.download = `studio_${job.id}.jpg`
  a.click()
}

/**
 * Trigger a browser download for all completed jobs.
 */
export function downloadAll(jobs: Job[]): void {
  const completed = jobs.filter((j) => j.status === 'completed' && j.resultImageUrl)
  completed.forEach((j, i) => {
    // Stagger downloads slightly so the browser doesn't block them
    setTimeout(() => downloadResult(j), i * 300)
  })
}
