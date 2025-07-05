import { useEffect, useState } from 'react'
import { useEventSource } from '../lib/useEventSource'
import { registerWidget } from '../lib/widgetRegistry'

interface SimEvent {
  data?: {
    world_map?: {
      agents?: Record<string, [number, number]>
    }
  }
}

export default function LiveMap() {
  const event = useEventSource<SimEvent>()
  const [positions, setPositions] = useState<Record<string, [number, number]>>({})
  const [summaries, setSummaries] = useState<string[]>([])

  const agentId = Object.keys(positions)[0] || 'agent-1'

  useEffect(() => {
    if (event?.data?.world_map?.agents) {
      setPositions(event.data.world_map.agents)
    }
  }, [event])

  useEffect(() => {
    let cancelled = false
    async function fetchSummaries() {
      try {
        const res = await fetch(`/api/agents/${agentId}/semantic_summaries`)
        const json = await res.json()
        if (!cancelled) setSummaries(json.summaries || [])
      } catch {
        /* ignore */
      }
    }
    fetchSummaries()
    return () => {
      cancelled = true
    }
  }, [agentId])

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Live Map</h1>
      <div data-testid="map-display">
        <ul>
          {Object.entries(positions).map(([id, pos]) => (
            <li key={id}>
              {id}: {pos[0]}, {pos[1]}
            </li>
          ))}
        </ul>
      </div>
      <div data-testid="summaries">
        {summaries.map((s, i) => (
          <div key={i}>{s}</div>
        ))}
      </div>
    </div>
  )
}

registerWidget('LiveMap', LiveMap)
