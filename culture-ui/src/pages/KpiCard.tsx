import KpiCard from '../widgets/KpiCard'
import { registerWidget } from '../lib/widgetRegistry'

export default function KpiCardPage() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">KPI Card</h1>
      <KpiCard />
    </div>
  )
}

registerWidget('KpiCard', KpiCardPage)

