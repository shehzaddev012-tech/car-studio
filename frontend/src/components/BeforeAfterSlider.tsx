/**
 * Before/after comparison slider.
 *
 * Shows the original image on the left and the studio result on the right.
 * A draggable divider lets the user reveal either side.
 * Pure CSS + pointer events — no external library.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { clsx } from 'clsx'

interface Props {
  beforeUrl: string
  afterUrl: string
  beforeLabel?: string
  afterLabel?: string
  className?: string
}

export function BeforeAfterSlider({
  beforeUrl,
  afterUrl,
  beforeLabel = 'Original',
  afterLabel = 'Studio',
  className,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [position, setPosition] = useState(50) // percentage 0–100
  const isDragging = useRef(false)

  const updatePosition = useCallback((clientX: number) => {
    const el = containerRef.current
    if (!el) return
    const { left, width } = el.getBoundingClientRect()
    const pct = Math.max(0, Math.min(100, ((clientX - left) / width) * 100))
    setPosition(pct)
  }, [])

  // Mouse
  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    isDragging.current = true
  }

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return
      updatePosition(e.clientX)
    }
    const onMouseUp = () => { isDragging.current = false }
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }
  }, [updatePosition])

  // Touch
  const onTouchStart = () => { isDragging.current = true }
  const onTouchMove = (e: React.TouchEvent) => {
    if (!isDragging.current) return
    updatePosition(e.touches[0].clientX)
  }
  const onTouchEnd = () => { isDragging.current = false }

  return (
    <div
      ref={containerRef}
      className={clsx('relative select-none overflow-hidden rounded-lg bg-studio-900', className)}
      style={{ cursor: isDragging.current ? 'ew-resize' : 'default' }}
    >
      {/* After (studio) — full width base layer */}
      <img
        src={afterUrl}
        alt="Studio result"
        className="block h-full w-full object-contain"
        draggable={false}
      />

      {/* Before (original) — clipped on the left side */}
      <div
        className="pointer-events-none absolute inset-0 overflow-hidden"
        style={{ width: `${position}%` }}
      >
        <img
          src={beforeUrl}
          alt="Original photo"
          className="block h-full w-full object-contain"
          style={{ width: `${(100 / position) * 100}%`, maxWidth: 'none' }}
          draggable={false}
        />
      </div>

      {/* Divider */}
      <div
        className="absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.2)] cursor-ew-resize"
        style={{ left: `${position}%`, transform: 'translateX(-50%)' }}
        onMouseDown={onMouseDown}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
      >
        {/* Handle */}
        <div className="absolute left-1/2 top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white shadow-lg ring-1 ring-studio-200">
          <svg className="h-4 w-4 text-studio-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 9l-4 3 4 3M16 9l4 3-4 3" />
          </svg>
        </div>
      </div>

      {/* Labels */}
      <div className="pointer-events-none absolute bottom-2 left-3">
        <span className="rounded bg-black/50 px-2 py-0.5 text-xs font-medium text-white backdrop-blur-sm">
          {beforeLabel}
        </span>
      </div>
      <div className="pointer-events-none absolute bottom-2 right-3">
        <span className="rounded bg-black/50 px-2 py-0.5 text-xs font-medium text-white backdrop-blur-sm">
          {afterLabel}
        </span>
      </div>
    </div>
  )
}
