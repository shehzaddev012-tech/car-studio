/**
 * WebSocket hook for real-time job status updates.
 *
 * Connects to /ws/jobs on mount, subscribes to job IDs as they appear,
 * and applies status updates to the React Query cache via useApplyWebSocketUpdate.
 *
 * Auto-reconnects on disconnect with exponential back-off (max 30 s).
 * Falls back gracefully if WebSocket is unavailable — React Query polling
 * (every 3 s in useJobsQuery) acts as the safety net.
 */
import { useCallback, useEffect, useRef } from 'react'
import type { JobProgressEvent } from '@/types'
import { useJobStore } from '@/store/jobStore'
import { useApplyWebSocketUpdate } from './useJobs'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/jobs`
const MAX_BACKOFF_MS = 30_000

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const backoffRef = useRef<number>(1_000)
  const subscribedRef = useRef<Set<string>>(new Set())

  const jobIds = useJobStore((s) => s.jobIds)
  const applyUpdate = useApplyWebSocketUpdate()

  const subscribe = useCallback((ids: string[]) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const newIds = ids.filter((id) => !subscribedRef.current.has(id))
    if (newIds.length === 0) return
    ws.send(JSON.stringify({ subscribe: newIds }))
    newIds.forEach((id) => subscribedRef.current.add(id))
  }, [])

  const connect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)

    let ws: WebSocket
    try {
      ws = new WebSocket(WS_URL)
    } catch {
      return // WebSocket not supported / blocked — fall back to polling
    }

    wsRef.current = ws

    ws.onopen = () => {
      backoffRef.current = 1_000 // reset back-off on successful connect
      subscribedRef.current.clear()
      // Subscribe to all currently tracked job IDs
      if (jobIds.length > 0) subscribe(jobIds)
    }

    ws.onmessage = (ev: MessageEvent<string>) => {
      if (ev.data === 'pong') return
      try {
        const event = JSON.parse(ev.data) as JobProgressEvent
        applyUpdate(event)
      } catch {
        // malformed message — ignore
      }
    }

    ws.onclose = () => {
      wsRef.current = null
      // Exponential back-off reconnect
      reconnectTimer.current = setTimeout(() => {
        backoffRef.current = Math.min(backoffRef.current * 1.5, MAX_BACKOFF_MS)
        connect()
      }, backoffRef.current)
    }

    ws.onerror = () => {
      ws.close() // triggers onclose → reconnect
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Initial connect
  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  // Subscribe whenever new job IDs are added to the store
  useEffect(() => {
    subscribe(jobIds)
  }, [jobIds, subscribe])

  // Heartbeat ping every 25 s to keep the connection alive
  useEffect(() => {
    const id = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping')
      }
    }, 25_000)
    return () => clearInterval(id)
  }, [])
}
