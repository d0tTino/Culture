import { useEffect, useState } from 'react'

export default function AgentMemories() {
  const [agentId, setAgentId] = useState('agent-1')
  const [summaries, setSummaries] = useState<string[]>([])

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch(`/api/agents/${agentId}/semantic_summaries`)
        const json = await res.json()
        if (!cancelled) setSummaries(json.summaries || [])
      } catch {
        /* ignore */
      }
    }
    void load()

    const es = new EventSource(`/api/agents/${agentId}/semantic_summaries`)
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as { summaries?: string[] }
        if (data.summaries) setSummaries(data.summaries)
      } catch {
        // ignore
      }
    }

    return () => {
      cancelled = true
      es.close()
    }
  }, [agentId])

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Agent Memories</h1>
      <label className="block">
        Agent ID:
        <input
          aria-label="agent-select"
          value={agentId}
          onChange={(e) => setAgentId(e.target.value)}
          className="border p-1 ml-2"
        />
      </label>
      <div data-testid="agent-memories">
        {summaries.map((s, i) => (
          <div key={i}>{s}</div>
        ))}
      </div>
    </div>
  )
}
