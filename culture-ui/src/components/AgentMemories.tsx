import { useAgentStream } from '../lib/useAgentStream'

export default function AgentMemories() {
  const agents = useAgentStream()?.agents ?? []

  return (
    <div>
      <h2 className="font-bold">Top Memories</h2>
      {agents.map((ag) => (
        <div key={ag.agent_id} className="mb-2">
          <h3 className="font-semibold">{ag.agent_id}</h3>
          <ul className="list-disc ml-4">
            {(ag.memories ?? []).map((m, i) => (
              <li key={i}>{m}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}
