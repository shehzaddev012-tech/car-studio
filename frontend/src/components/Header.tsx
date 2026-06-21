interface Props {
  completedCount: number
  totalCount: number
}

export function Header({ completedCount, totalCount }: Props) {
  return (
    <header className="border-b border-studio-200 bg-white">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-studio-900">
              <svg
                className="h-4 w-4 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"
                />
                <polyline
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points="9 22 9 12 15 12 15 22"
                />
              </svg>
            </div>
            <div>
              <span className="text-sm font-semibold text-studio-900">Car Studio</span>
              <span className="ml-1.5 text-sm text-studio-500">AI Background Generator</span>
            </div>
          </div>

          {/* Stats */}
          {totalCount > 0 && (
            <div className="flex items-center gap-4 text-sm text-studio-600">
              <span>
                <span className="font-semibold text-emerald-600">{completedCount}</span>
                <span className="mx-1">/</span>
                <span className="font-medium">{totalCount}</span>
                <span className="ml-1">completed</span>
              </span>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
