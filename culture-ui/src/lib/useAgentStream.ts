import { useEffect, useState } from 'react'

export interface AgentData {
  agent_id: string
  state?: Record<string, unknown>
  mood?: number | null
  memories?: string[]
}

interface AgentStreamPayload {
  agents: AgentData[]
}

export function useAgentStream() {
  const [data, setData] = useState<AgentStreamPayload | null>(null)

  useEffect(() => {
    const es = new EventSource('/stream/agents')
    es.onmessage = (ev) => {
      try {
        setData(JSON.parse(ev.data) as AgentStreamPayload)
      } catch {
        // ignore parse errors
      }
    }
    return () => es.close()
  }, [])

  return data
}
