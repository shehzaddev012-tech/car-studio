/**
 * Zustand store for client-side UI state.
 *
 * Manages:
 *   - jobIds: ordered list of job IDs created this session (drives polling scope)
 *   - pendingUploads: local file previews before upload completes
 *   - uploadState: idle | uploading | error
 *
 * Server state (job data, status) lives in React Query — not here.
 */
import { create } from 'zustand'
import type { Job, PendingUpload, UploadState } from '@/types'

interface JobStoreState {
  /** Ordered list of all job IDs created this session. */
  jobIds: string[]
  /** Client-side file previews before upload starts. */
  pendingUploads: PendingUpload[]
  uploadState: UploadState
  uploadError: string | null

  // Actions
  addJobs: (jobs: Job[]) => void
  clearPendingUploads: () => void
  addPendingUploads: (uploads: PendingUpload[]) => void
  setUploadState: (state: UploadState, error?: string) => void
}

export const useJobStore = create<JobStoreState>((set) => ({
  jobIds: [],
  pendingUploads: [],
  uploadState: 'idle',
  uploadError: null,

  addJobs: (jobs) =>
    set((s) => ({
      jobIds: [
        ...s.jobIds,
        ...jobs.map((j) => j.id).filter((id) => !s.jobIds.includes(id)),
      ],
      pendingUploads: [],
      uploadState: 'idle',
    })),

  clearPendingUploads: () =>
    set({ pendingUploads: [], uploadState: 'idle', uploadError: null }),

  addPendingUploads: (uploads) =>
    set((s) => ({
      pendingUploads: [...s.pendingUploads, ...uploads],
    })),

  setUploadState: (state, error) =>
    set({ uploadState: state, uploadError: error ?? null }),
}))
