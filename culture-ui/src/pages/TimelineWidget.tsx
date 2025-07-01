import TimelineWidget from '../widgets/TimelineWidget'
import { registerWidget } from '../lib/widgetRegistry'

export default function TimelineWidgetPage() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Timeline Widget</h1>
      <TimelineWidget />
    </div>
  )
}

registerWidget('TimelineWidget', TimelineWidgetPage)

