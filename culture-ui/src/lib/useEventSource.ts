import { useEffect, useState } from 'react'

export interface EventSourceMessage<T = unknown> {
  data: T
}

/**
 * Hook that connects to `/stream/events` via EventSource and
 * optionally falls back to WebSocket when SSE is unavailable.
 * Returns the latest parsed event data or `null` if no event has been
 * received yet.
 */
export function useEventSource<T = unknown>() {
  const [event, setEvent] = useState<T | null>(null)

  useEffect(() => {
    let es: EventSource | null = null
    let ws: WebSocket | null = null
    let active = true

    const handleMessage = (data: string) => {
      try {
        setEvent(JSON.parse(data))
      } catch {
        // not JSON, return raw string
        setEvent(data as unknown as T)
      }
    }

    const connectWebSocket = () => {
      if (typeof WebSocket === 'undefined') return
      try {
        ws = new WebSocket('/ws/events')
        ws.onmessage = (ev) => active && handleMessage(ev.data)
      } catch {
        // ignore
      }
    }

    if (typeof EventSource !== 'undefined') {
      try {
        es = new EventSource('/stream/events')
        es.onmessage = (ev) => active && handleMessage(ev.data)
        es.onerror = () => {
          es?.close()
          connectWebSocket()
        }
      } catch {
        connectWebSocket()
      }
    } else {
      connectWebSocket()
    }

    return () => {
      active = false
      es?.close()
      ws?.close()
    }
  }, [])

  return event
}
