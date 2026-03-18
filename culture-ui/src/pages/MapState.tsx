import { useEffect, useState } from 'react'
import { fetchApiEnvelope } from '../lib/api'
import { useCapabilities } from '../lib/useCapabilities'

export default function MapState() {
  const capabilities = useCapabilities()
  const mapCapability = capabilities.map
  const [positions, setPositions] = useState<Record<string, [number, number]>>({})

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const envelope = await fetchApiEnvelope<{ world_map?: { agents?: Record<string, [number, number]> } }>('/api/map')
        if (!cancelled && envelope.data.world_map?.agents) {
          setPositions(envelope.data.world_map.agents)
        }
      } catch {
        if (!cancelled) {
          setPositions({})
        }
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Map State</h1>
      {mapCapability && !mapCapability.enabled && <p data-testid="map-fallback">{mapCapability.reason}</p>}
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
