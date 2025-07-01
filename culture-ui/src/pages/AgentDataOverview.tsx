import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts'
import { agentMetrics } from '../mock/agentMetrics'
import { registerWidget } from '../lib/widgetRegistry'

export default function AgentDataOverview() {
  const currentAgents = agentMetrics[agentMetrics.length - 1].activeAgents
  const totalMessages = agentMetrics.reduce((sum, m) => sum + m.messages, 0)
  const avgAgents = Math.round(
    agentMetrics.reduce((sum, m) => sum + m.activeAgents, 0) / agentMetrics.length,
  )

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Agent Data Overview</h1>
      <div className="flex space-x-4" data-testid="kpi-metrics">
        <div className="border rounded p-2">
          <div className="text-sm">Active Agents</div>
          <div className="text-lg font-bold" data-testid="active-agents">{currentAgents}</div>
        </div>
        <div className="border rounded p-2">
          <div className="text-sm">Total Messages</div>
          <div className="text-lg font-bold" data-testid="total-messages">{totalMessages}</div>
        </div>
        <div className="border rounded p-2">
          <div className="text-sm">Avg Agents</div>
          <div className="text-lg font-bold" data-testid="avg-agents">{avgAgents}</div>
        </div>
      </div>
      <div style={{ width: 400, height: 300 }} data-testid="agents-line-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={agentMetrics} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <XAxis dataKey="date" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Line type="monotone" dataKey="activeAgents" stroke="#8884d8" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

registerWidget('agentData', AgentDataOverview)
