import { useEffect, useState } from 'react'

interface MapEvent {
  type: string
  data?: {
    world_map?: {
      agents?: Record<string, [number, number]>
    }
    agents?: Record<string, { mood?: number; summary?: string }>
  }
}

export default function LiveMap() {
  const [positions, setPositions] = useState<Record<string, [number, number]>>({})
  const [moods, setMoods] = useState<Record<string, number>>({})
  const [summaries, setSummaries] = useState<Record<string, string>>({})

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch('/api/map')
        const json = (await res.json()) as {
          world_map?: { agents?: Record<string, [number, number]> }
          agents?: Record<string, { mood?: number; summary?: string }>
        }
        if (cancelled) return
        if (json.world_map?.agents) setPositions(json.world_map.agents)
        if (json.agents) {
          const m: Record<string, number> = {}
          const s: Record<string, string> = {}
          for (const [id, info] of Object.entries(json.agents)) {
            if (typeof info.mood === 'number') m[id] = info.mood
            if (info.summary) s[id] = info.summary
          }
          setMoods(m)
          setSummaries(s)
        }
      } catch {
        /* ignore */
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const es = new EventSource('/api/map/stream')
    es.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data) as MapEvent
        if (payload.data?.world_map?.agents) {
          setPositions(payload.data.world_map.agents)
        }
        if (payload.data?.agents) {
          setMoods((cur) => {
            const copy = { ...cur }
            for (const [id, info] of Object.entries(payload.data!.agents!)) {
              if (typeof info.mood === 'number') copy[id] = info.mood
            }
            return copy
          })
          setSummaries((cur) => {
            const copy = { ...cur }
            for (const [id, info] of Object.entries(payload.data!.agents!)) {
              if (info.summary) copy[id] = info.summary
            }
            return copy
          })
        }
      } catch {
        /* ignore */
      }
    }
    return () => es.close()
  }, [])

  return (
    <div data-testid="live-map">
      <ul>
        {Object.entries(positions).map(([id, [x, y]]) => (
          <li key={id}>
            {id}: {x}, {y} (mood {moods[id] ?? 'n/a'}) - {summaries[id] ?? ''}
          </li>
        ))}
      </ul>
    </div>
  )
}
