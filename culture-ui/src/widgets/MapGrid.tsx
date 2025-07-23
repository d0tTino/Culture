import { useEffect, useState } from 'react'
import { useEventSource } from '../lib/useEventSource'

interface MapData {
  width: number
  height: number
  agents?: Record<string, [number, number]>
}

interface MapEvent {
  data?: { world_map?: MapData }
}

export default function MapGrid() {
  const [map, setMap] = useState<MapData | null>(null)
  const event = useEventSource<MapEvent>()

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch('/api/map')
        const json = (await res.json()) as { world_map: MapData }
        if (!cancelled) setMap(json.world_map)
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
    if (event?.data?.world_map) setMap(event.data.world_map)
  }, [event])

  if (!map) return <div data-testid="map-grid" />

  const cells = []
  for (let y = 0; y < map.height; y++) {
    for (let x = 0; x < map.width; x++) {
      const agent = Object.entries(map.agents ?? {}).find(([, pos]) => pos[0] === x && pos[1] === y)?.[0]
      cells.push(
        <div
          key={`${x}-${y}`}
          data-testid={`cell-${x}-${y}`}
          style={{ width: 20, height: 20, border: '1px solid black', boxSizing: 'border-box' }}
        >
          {agent && <span data-testid={`agent-${agent}`}>{agent}</span>}
        </div>
      )
    }
  }

  return (
    <div data-testid="map-grid" style={{ display: 'grid', gridTemplateColumns: `repeat(${map.width}, 20px)` }}>
      {cells}
    </div>
  )
}
