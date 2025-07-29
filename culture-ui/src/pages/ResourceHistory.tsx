import ResourceHistory from '../widgets/ResourceHistory'
import { registerWidget } from '../lib/widgetRegistry'

export default function ResourceHistoryPage() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Resource History</h1>
      <ResourceHistory agentId="agent-1" />
    </div>
  )
}

registerWidget('ResourceHistory', ResourceHistoryPage)
