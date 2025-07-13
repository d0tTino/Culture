import { useEffect, useState } from 'react'
import { useEventSource } from '../lib/useEventSource'
import { registerWidget } from '../lib/widgetRegistry'

interface WorldMapData {
  agents?: Record<string, [number, number]>
  resources?: Record<string, Record<string, number>>
}

interface SimEvent {
  data?: { world_map?: WorldMapData }
}

export default function WorldMapPage() {
  const event = useEventSource<SimEvent>()
  const [map, setMap] = useState<WorldMapData>({})

  useEffect(() => {
    if (event?.data?.world_map) setMap(event.data.world_map)
  }, [event])

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">World Map</h1>
      <div data-testid="world-map">
        <ul>
          {Object.entries(map.agents ?? {}).map(([id, [x, y]]) => (
            <li key={id}>
              {id}: {x}, {y}
            </li>
          ))}
        </ul>
        <ul>
          {Object.entries(map.resources ?? {}).map(([pos, res]) => (
            <li key={pos}>
              {pos}: {Object.entries(res)
                .map(([r, a]) => `${r}:${a}`)
                .join(', ')}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

registerWidget('WorldMap', WorldMapPage)
