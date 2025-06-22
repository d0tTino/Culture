import data from '../mock/agentKpi.json'
import { ResponsiveContainer, LineChart, Line } from 'recharts'

interface Metric {
  label: string
  value: number
}

interface KpiData {
  metrics: Metric[]
  sparkline: number[]
}

const kpiData = data as KpiData

export default function AgentDataOverview() {
  return (
    <div className="p-4">
      <h2 className="text-2xl font-semibold mb-4">Agent Data Overview</h2>
      <div className="grid gap-4 sm:grid-cols-3">
        {kpiData.metrics.map((metric) => (
          <div key={metric.label} className="rounded-lg bg-card p-4 shadow">
            <div className="text-sm text-muted-foreground">{metric.label}</div>
            <div className="text-2xl font-bold">{metric.value}</div>
          </div>
        ))}
      </div>
      <div className="mt-6 h-24">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={kpiData.sparkline.map((v, i) => ({ index: i, value: v }))}>
            <Line
              type="monotone"
              dataKey="value"
              stroke="var(--color-chart-1)"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
