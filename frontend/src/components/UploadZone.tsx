/**
 * Drag-and-drop upload zone with instant local previews.
 *
 * Validates client-side before upload:
 *   - Accepted MIME types: image/jpeg, image/png, image/webp
 *   - Max size: 25 MB per file
 *
 * Shows thumbnail previews immediately (before upload) using
 * URL.createObjectURL so there's no waiting.
 */
import { useCallback, useRef, useState } from 'react'
import { clsx } from 'clsx'
import { useCreateJobs } from '@/hooks/useJobs'
import { useJobStore } from '@/store/jobStore'
import type { PendingUpload } from '@/types'

const ACCEPTED_TYPES = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp'])
const MAX_SIZE_MB = 25
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1_048_576

function validateFile(file: File): string | null {
  if (!ACCEPTED_TYPES.has(file.type)) return `"${file.name}" is not a supported image type (JPEG, PNG, WebP only).`
  if (file.size > MAX_SIZE_BYTES) return `"${file.name}" exceeds the ${MAX_SIZE_MB} MB limit.`
  if (file.size < 1_024) return `"${file.name}" appears to be empty or corrupt.`
  return null
}

export function UploadZone() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [validationErrors, setValidationErrors] = useState<string[]>([])

  const { mutateAsync: createJobs, isPending: isUploading } = useCreateJobs()
  const { pendingUploads, addPendingUploads, clearPendingUploads, setUploadState, uploadError } =
    useJobStore()

  const handleFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return

      // Client-side validation
      const errors: string[] = []
      const valid: File[] = []
      for (const f of files) {
        const err = validateFile(f)
        if (err) errors.push(err)
        else valid.push(f)
      }
      setValidationErrors(errors)
      if (valid.length === 0) return

      // Instant local previews
      const uploads: PendingUpload[] = valid.map((f) => ({
        file: f,
        previewUrl: URL.createObjectURL(f),
      }))
      addPendingUploads(uploads)
      setUploadState('uploading')

      try {
        await createJobs(valid)
        // On success: addJobs() in the mutation clears pendingUploads
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Upload failed. Please try again.'
        setUploadState('error', msg)
        // Clean up preview object URLs on failure
        uploads.forEach((u) => URL.revokeObjectURL(u.previewUrl))
        clearPendingUploads()
      }
    },
    [addPendingUploads, clearPendingUploads, createJobs, setUploadState],
  )

  // ── Drag events ────────────────────────────────────────────────────────────

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }
  const onDragLeave = () => setIsDragging(false)
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    void handleFiles(files)
  }

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    void handleFiles(files)
    // Reset so the same file can be re-selected
    e.target.value = ''
  }

  // ── Pending preview strip ──────────────────────────────────────────────────

  const hasPreviews = pendingUploads.length > 0

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload car photos"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={clsx(
          'relative flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2',
          isDragging
            ? 'border-blue-400 bg-blue-50'
            : 'border-studio-300 bg-studio-50 hover:border-studio-400 hover:bg-studio-100',
        )}
      >
        {isUploading ? (
          <div className="flex flex-col items-center gap-3 py-8">
            <svg className="h-8 w-8 animate-spin text-blue-500" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <p className="text-sm font-medium text-studio-700">Uploading…</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 px-6 py-10 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-studio-200">
              <svg className="h-6 w-6 text-studio-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-semibold text-studio-800">
                {isDragging ? 'Drop photos here' : 'Drop car photos here'}
              </p>
              <p className="mt-1 text-xs text-studio-500">
                or <span className="font-medium text-blue-600">browse files</span> · JPEG, PNG, WebP up to {MAX_SIZE_MB} MB
              </p>
            </div>
            <p className="text-xs text-studio-400">Multiple files supported</p>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          multiple
          accept="image/jpeg,image/jpg,image/png,image/webp"
          className="sr-only"
          onChange={onInputChange}
        />
      </div>

      {/* Validation errors */}
      {validationErrors.length > 0 && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3">
          <p className="mb-1 text-xs font-semibold text-red-700">Some files were rejected:</p>
          <ul className="space-y-0.5">
            {validationErrors.map((e, i) => (
              <li key={i} className="text-xs text-red-600">
                {e}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Upload error from server */}
      {uploadError && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3">
          <p className="text-xs font-semibold text-red-700">Upload failed</p>
          <p className="text-xs text-red-600">{uploadError}</p>
        </div>
      )}

      {/* Pending previews */}
      {hasPreviews && (
        <div>
          <p className="mb-2 text-xs font-medium text-studio-500 uppercase tracking-wide">Uploading…</p>
          <div className="flex flex-wrap gap-3">
            {pendingUploads.map((u, i) => (
              <div key={i} className="relative h-20 w-20 overflow-hidden rounded-lg bg-studio-100 ring-1 ring-studio-200">
                <img
                  src={u.previewUrl}
                  alt={u.file.name}
                  className="h-full w-full object-cover"
                />
                <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                  <svg className="h-5 w-5 animate-spin text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
