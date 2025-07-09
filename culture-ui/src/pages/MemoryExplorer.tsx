import EventConsole from '../widgets/EventConsole'
import BreakpointList from '../widgets/BreakpointList'
import { useEffect, useState } from 'react'

export default function MemoryExplorer() {
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
    return () => {
      cancelled = true
    }
  }, [agentId])

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Memory Explorer</h1>
      <label className="block">
        Agent ID:
        <input
          aria-label="agent-select"
          value={agentId}
          onChange={(e) => setAgentId(e.target.value)}
          className="border p-1 ml-2"
        />
      </label>
      <div className="grid gap-4 grid-cols-2">
        <BreakpointList />
        <EventConsole />
      </div>
      <div data-testid="summaries">
        {summaries.map((s, i) => (
          <div key={i}>{s}</div>
        ))}
      </div>
    </div>
  )
}

