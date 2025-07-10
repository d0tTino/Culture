import AgentTimeline from '../widgets/AgentTimeline'
import { registerWidget } from '../lib/widgetRegistry'

export default function AgentTimelinePage() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Agent Timeline</h1>
      <AgentTimeline />
    </div>
  )
}

registerWidget('AgentTimeline', AgentTimelinePage)

