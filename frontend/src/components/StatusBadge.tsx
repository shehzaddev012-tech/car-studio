import type { JobStatus } from '@/types'
import { clsx } from 'clsx'

const CONFIG: Record<JobStatus, { label: string; classes: string; dot: string }> = {
  pending: {
    label: 'Pending',
    classes: 'bg-studio-100 text-studio-700 border-studio-200',
    dot: 'bg-studio-400',
  },
  processing: {
    label: 'Processing',
    classes: 'bg-blue-50 text-blue-700 border-blue-200',
    dot: 'bg-blue-500 animate-pulse',
  },
  completed: {
    label: 'Completed',
    classes: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    dot: 'bg-emerald-500',
  },
  failed: {
    label: 'Failed',
    classes: 'bg-red-50 text-red-700 border-red-200',
    dot: 'bg-red-500',
  },
}

interface Props {
  status: JobStatus
  className?: string
}

export function StatusBadge({ status, className }: Props) {
  const { label, classes, dot } = CONFIG[status]
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium',
        classes,
        className,
      )}
    >
      <span className={clsx('h-1.5 w-1.5 rounded-full', dot)} />
      {label}
    </span>
  )
}
