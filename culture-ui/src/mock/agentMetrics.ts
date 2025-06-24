export interface AgentMetric {
  date: string
  activeAgents: number
  messages: number
}

export const agentMetrics: AgentMetric[] = [
  { date: '2024-05-01', activeAgents: 5, messages: 10 },
  { date: '2024-05-02', activeAgents: 7, messages: 20 },
  { date: '2024-05-03', activeAgents: 6, messages: 15 },
  { date: '2024-05-04', activeAgents: 8, messages: 25 },
  { date: '2024-05-05', activeAgents: 9, messages: 30 },
]
