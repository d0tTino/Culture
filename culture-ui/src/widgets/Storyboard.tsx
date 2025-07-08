import { useEffect, useState } from 'react'
import { useEventSource } from '../lib/useEventSource'

interface SnapshotEvent {
  data?: {
    world_map?: { agents?: Record<string, [number, number]> }
    agents?: Array<{ agent_id: string; mood?: number }>
  }
}

export default function Storyboard() {
  const event = useEventSource<SnapshotEvent>()
  const [positions, setPositions] = useState<Record<string, [number, number]>>({})
  const [moods, setMoods] = useState<Record<string, number>>({})
  const [tab, setTab] = useState<'map' | 'summaries'>('map')
  const [summaries, setSummaries] = useState<string[]>([])

  const agentId = Object.keys(positions)[0] || Object.keys(moods)[0] || 'agent-1'

  useEffect(() => {
    if (event?.data?.world_map?.agents) {
      setPositions(event.data.world_map.agents)
    }
    if (event?.data?.agents) {
      const m: Record<string, number> = {}
      for (const a of event.data.agents) {
        if (typeof a.mood === 'number') m[a.agent_id] = a.mood
      }
      setMoods((cur) => ({ ...cur, ...m }))
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
        <ul data-testid="map-info">
          {Object.entries(positions).map(([id, pos]) => (
            <li key={id}>
              {id}: {pos[0]}, {pos[1]} (mood {moods[id] ?? 'n/a'})
            </li>
          ))}
        </ul>
      ) : (
        <div data-testid="summaries">
          {summaries.map((s, i) => (
            <div key={i}>{s}</div>
          ))}
        </div>
      )}
    </div>
  )
}
