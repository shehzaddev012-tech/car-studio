// ── Domain types ──────────────────────────────────────────────────────────────

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed'

export interface Job {
  id: string
  status: JobStatus
  originalImageUrl: string
  resultImageUrl?: string
  errorMessage?: string
  progressPercent?: number
  createdAt: string
  updatedAt: string
}

// ── API response shapes ────────────────────────────────────────────────────────

export interface JobListResponse {
  jobs: Job[]
  total: number
}

// ── WebSocket event payload ────────────────────────────────────────────────────

export interface JobProgressEvent {
  jobId: string
  status: JobStatus
  progressPercent?: number
  message?: string
  resultImageUrl?: string
  errorMessage?: string
}

// ── Upload state (client-side, before the job is persisted) ──────────────────

export interface PendingUpload {
  /** Browser-local preview URL (from URL.createObjectURL) */
  previewUrl: string
  file: File
}

// ── UI state atoms ─────────────────────────────────────────────────────────────

export type UploadState = 'idle' | 'uploading' | 'error'
