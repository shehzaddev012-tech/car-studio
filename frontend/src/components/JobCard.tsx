/**
 * Individual job card.
 *
 * States:
 *   pending    — thumbnail + "Queued" state
 *   processing — thumbnail + progress bar + step message
 *   completed  — before/after comparison slider (expandable)
 *   failed     — error message + retry hint
 */
import { useState } from 'react'
import { clsx } from 'clsx'
import { StatusBadge } from './StatusBadge'
import { BeforeAfterSlider } from './BeforeAfterSlider'
import { downloadResult } from '@/api/client'
import { useCancelJob } from '@/hooks/useJobs'
import type { Job } from '@/types'

interface Props {
  job: Job
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function JobCard({ job }: Props) {
  const [expanded, setExpanded] = useState(false)
  const { mutate: cancel, isPending: isCancelling } = useCancelJob()

  const isCompleted = job.status === 'completed'
  const isProcessing = job.status === 'processing'
  const isFailed = job.status === 'failed'
  const isPending = job.status === 'pending'

  return (
    <div
      className={clsx(
        'flex flex-col overflow-hidden rounded-xl border bg-white shadow-card transition-shadow duration-150',
        isCompleted && 'border-emerald-200',
        isProcessing && 'border-blue-200',
        isFailed && 'border-red-200',
        isPending && 'border-studio-200',
      )}
    >
      {/* ── Status stripe ─────────────────────────────────────────────────── */}
      <div
        className={clsx(
          'h-1 w-full',
          isCompleted && 'bg-emerald-400',
          isProcessing && 'bg-blue-400',
          isFailed && 'bg-red-400',
          isPending && 'bg-studio-300',
        )}
      />

      {/* ── Image area ────────────────────────────────────────────────────── */}
      {isCompleted && job.resultImageUrl ? (
        <div>
          {expanded ? (
            <BeforeAfterSlider
              beforeUrl={job.originalImageUrl}
              afterUrl={job.resultImageUrl}
              className="aspect-video w-full"
            />
          ) : (
            <button
              type="button"
              className="relative block w-full overflow-hidden bg-studio-100"
              onClick={() => setExpanded(true)}
              title="Click to compare before/after"
            >
              <img
                src={job.resultImageUrl}
                alt="Studio result"
                className="aspect-video w-full object-cover"
              />
              <div className="absolute inset-0 flex items-center justify-center bg-black/0 transition-colors hover:bg-black/20">
                <span className="rounded-full bg-black/50 px-3 py-1 text-xs font-medium text-white opacity-0 backdrop-blur-sm transition-opacity hover:opacity-100 group-hover:opacity-100">
                  Compare
                </span>
              </div>
              <div className="absolute right-2 top-2">
                <span className="rounded bg-black/50 px-2 py-0.5 text-[10px] font-medium text-white backdrop-blur-sm">
                  Click to compare
                </span>
              </div>
            </button>
          )}
        </div>
      ) : (
        <div className="relative aspect-video w-full overflow-hidden bg-studio-100">
          <img
            src={job.originalImageUrl}
            alt="Original"
            className="h-full w-full object-cover"
          />
          {/* Overlay for non-completed states */}
          {(isProcessing || isPending) && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/30 backdrop-blur-[2px]">
              {isProcessing && (
                <svg className="h-8 w-8 animate-spin text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              )}
              {isPending && (
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-white/20">
                  <svg className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6l4 2m6-2a10 10 0 11-20 0 10 10 0 0120 0z" />
                  </svg>
                </div>
              )}
            </div>
          )}
          {isFailed && (
            <div className="absolute inset-0 flex items-center justify-center bg-red-900/30">
              <svg className="h-8 w-8 text-red-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          )}
        </div>
      )}

      {/* ── Card body ─────────────────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col gap-3 p-4">
        {/* Header row */}
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <StatusBadge status={job.status} />
            <p className="mt-1 truncate text-xs text-studio-400" title={job.id}>
              {job.id.slice(0, 8)}…
            </p>
          </div>
          <span className="shrink-0 text-xs text-studio-400">{formatTime(job.createdAt)}</span>
        </div>

        {/* Progress bar */}
        {isProcessing && (
          <div className="space-y-1">
            <div className="h-1.5 overflow-hidden rounded-full bg-studio-100">
              <div
                className="h-full rounded-full bg-blue-500 transition-all duration-500"
                style={{ width: `${job.progressPercent ?? 0}%` }}
              />
            </div>
            <p className="text-xs text-studio-500">
              {Math.round(job.progressPercent ?? 0)}% complete
            </p>
          </div>
        )}

        {/* Error message */}
        {isFailed && job.errorMessage && (
          <div className="rounded-lg bg-red-50 p-2.5">
            <p className="text-xs font-medium text-red-700">Why it failed:</p>
            <p className="mt-0.5 text-xs leading-relaxed text-red-600">{job.errorMessage}</p>
          </div>
        )}

        {/* Action buttons */}
        <div className="mt-auto flex gap-2">
          {isCompleted && job.resultImageUrl && (
            <>
              <button
                type="button"
                onClick={() => setExpanded((v) => !v)}
                className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-studio-200 bg-white px-3 py-2 text-xs font-medium text-studio-700 transition-colors hover:bg-studio-50"
              >
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
                {expanded ? 'Collapse' : 'Compare'}
              </button>
              <button
                type="button"
                onClick={() => downloadResult(job)}
                className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-studio-900 px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-studio-800"
              >
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download
              </button>
            </>
          )}

          {isPending && (
            <button
              type="button"
              disabled={isCancelling}
              onClick={() => cancel(job.id)}
              className="flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs font-medium text-red-600 transition-colors hover:bg-red-100 disabled:opacity-50"
            >
              {isCancelling ? 'Cancelling…' : 'Cancel'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
