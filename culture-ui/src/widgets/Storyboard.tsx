import { useEffect, useState } from 'react'

interface SnapshotEvent {
  event_type?: string
  data?: {
    world_map?: { agents?: Record<string, [number, number]> }
    agents?: Array<{ agent_id: string; mood?: number }>
    step?: number
  }
}

export default function Storyboard() {
  const [event, setEvent] = useState<SnapshotEvent | null>(null)
  useEffect(() => {
    const ws = new WebSocket('/ws/events')
    ws.onmessage = (ev) => {
      try {
        setEvent(JSON.parse(ev.data))
      } catch {
        /* ignore parse errors */
      }
    }
    return () => {
      ws.close()
    }
  }, [])
  const [positions, setPositions] = useState<Record<string, [number, number]>>({})
  const [moods, setMoods] = useState<Record<string, number>>({})
  const [heatmap, setHeatmap] = useState<Record<string, number>>({})
  const [memoryEvents, setMemoryEvents] = useState<
    Array<{ type: string; step?: number }>
  >([])
  const [tab, setTab] = useState<'map' | 'summaries'>('map')
  const [summaries, setSummaries] = useState<string[]>([])

  const agentId = Object.keys(positions)[0] || Object.keys(moods)[0] || 'agent-1'

  useEffect(() => {
    if (event?.data?.world_map?.agents) {
      setPositions(event.data.world_map.agents)
      setHeatmap((cur) => {
        const copy = { ...cur }
        for (const [, pos] of Object.entries(event.data!.world_map!.agents!)) {
          const key = `${Math.round(pos[0])},${Math.round(pos[1])}`
          copy[key] = (copy[key] || 0) + 1
        }
        return copy
      })
    }
    if (event?.data?.agents) {
      const m: Record<string, number> = {}
      for (const a of event.data.agents) {
        if (typeof a.mood === 'number') m[a.agent_id] = a.mood
      }
      setMoods((cur) => ({ ...cur, ...m }))
    }
    if (event?.event_type?.startsWith('memory')) {
      setMemoryEvents((cur) =>
        [{ type: event.event_type!, step: event.data?.step }, ...cur].slice(0, 20),
      )
    }
  }, [event])

  useEffect(() => {
    if (tab !== 'summaries') return
    let cancelled = false
    async function load() {
      try {
        const res = await fetch(`/api/agents/${agentId}/semantic_summaries`)
        const json = await res.json()
        if (!cancelled) setSummaries(json.summaries || [])
      } catch {
        /* ignore */
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [tab, agentId])

  return (
    <div className="p-2" data-testid="storyboard">
      <div className="space-x-2 mb-2">
        <button onClick={() => setTab('map')}>Map</button>
        <button onClick={() => setTab('summaries')}>Summaries</button>
      </div>
      {tab === 'map' ? (
        <div>
          <ul data-testid="map-info" className="mb-2">
            {Object.entries(positions).map(([id, pos]) => (
              <li key={id}>
                {id}: {pos[0]}, {pos[1]} (mood {moods[id] ?? 'n/a'})
              </li>
            ))}
          </ul>
          <div
            data-testid="heatmap"
            className="grid grid-cols-10 grid-rows-10 w-40 h-40 border"
          >
            {Array.from({ length: 100 }).map((_, i) => {
              const x = i % 10
              const y = Math.floor(i / 10)
              const count = heatmap[`${x},${y}`] || 0
              const max = Math.max(1, ...Object.values(heatmap))
              const alpha = count / max
              return (
                <div
                  key={`${x}-${y}`}
                  style={{ backgroundColor: `rgba(255,0,0,${alpha})` }}
                />
              )
            })}
          </div>
        </div>
      ) : (
        <div data-testid="summaries">
          {summaries.map((s, i) => (
            <div key={i}>{s}</div>
          ))}
        </div>
      )}
      <div
        data-testid="memory-events"
        className="max-h-32 overflow-y-auto mt-2 text-sm"
      >
        {memoryEvents.map((e, i) => (
          <div key={i}>
            {e.type}
            {e.step !== undefined ? ` (step ${e.step})` : ''}
          </div>
        ))}
      </div>
    </div>
  )
}
