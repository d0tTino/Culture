import NetworkWeb from '../widgets/NetworkWeb'
import { registerWidget } from '../lib/widgetRegistry'

export default function NetworkWebPage() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Network Web</h1>
      <NetworkWeb />
    </div>
  )
}

registerWidget('NetworkWeb', NetworkWebPage)

