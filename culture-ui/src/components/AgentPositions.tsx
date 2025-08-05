import { useAgentStream } from '../lib/useAgentStream'

export default function AgentPositions() {
  const data = useAgentStream()
  const agents = data?.agents ?? []

  return (
    <div>
      <h2 className="font-bold">Agent Positions</h2>
      <ul>
        {agents.map((ag) => {
          const state = ag.state as Record<string, unknown> | undefined
          const pos = state?.position as { x: number; y: number } | undefined
          const text = pos ? `${pos.x}, ${pos.y}` : 'unknown'
          return (
            <li key={ag.agent_id}>
              {ag.agent_id}: {text}
            </li>
          )
        })}
      </ul>
    </div>
  )
}
