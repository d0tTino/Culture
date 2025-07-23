import { useEffect, useState } from 'react'

interface MapEvent {
  type?: string
  data?: {
    world_map?: {
      agents?: Record<string, [number, number]>
    }
  }
}

export default function MapState() {
  const [positions, setPositions] = useState<Record<string, [number, number]>>({})

  useEffect(() => {
    const es = new EventSource('/api/map')
    es.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data) as MapEvent
        if (payload.data?.world_map?.agents) {
          setPositions(payload.data.world_map.agents)
        }
      } catch {
        // ignore
      }
    }
    return () => es.close()
  }, [])

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Map State</h1>
      <div data-testid="map-state">
        <ul>
          {Object.entries(positions).map(([id, [x, y]]) => (
            <li key={id}>
              {id}: {x}, {y}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
