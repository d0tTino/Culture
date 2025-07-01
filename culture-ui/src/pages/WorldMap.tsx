import WorldMap from '../widgets/WorldMap'
import { registerWidget } from '../lib/widgetRegistry'

export default function WorldMapPage() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">World Map</h1>
      <WorldMap />
    </div>
  )
}

registerWidget('WorldMap', WorldMapPage)

