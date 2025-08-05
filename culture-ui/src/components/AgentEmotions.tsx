import { useAgentStream } from '../lib/useAgentStream'

export default function AgentEmotions() {
  const agents = useAgentStream()?.agents ?? []

  return (
    <div>
      <h2 className="font-bold">Agent Emotions</h2>
      <ul>
        {agents.map((ag) => (
          <li key={ag.agent_id}>
            {ag.agent_id}: {ag.mood ?? 'n/a'}
          </li>
        ))}
      </ul>
    </div>
  )
}
