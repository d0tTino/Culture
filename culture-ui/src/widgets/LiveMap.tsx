import { useEffect, useState } from 'react'

interface MapEvent {
  type: string
  data?: {
    world_map?: {
      agents?: Record<string, [number, number]>
    }
  }
}

export default function LiveMap() {
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
        // ignore invalid JSON
      }
    }
    return () => es.close()
  }, [])

  return (
    <div data-testid="live-map">
      <ul>
        {Object.entries(positions).map(([id, [x, y]]) => (
          <li key={id}>
            {id}: {x}, {y}
          </li>
        ))}
      </ul>
    </div>
  )
}
