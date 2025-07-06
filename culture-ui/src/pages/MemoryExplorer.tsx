import EventConsole from '../widgets/EventConsole'
import BreakpointList from '../widgets/BreakpointList'
import { useEffect, useState } from 'react'

export default function MemoryExplorer() {
  const [summaries, setSummaries] = useState<string[]>([])

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch('/api/agents/agent-1/semantic_summaries')
        const json = await res.json()
        setSummaries(json.summaries || [])
      } catch {
        /* ignore */
      }
    }
    void load()
  }, [])

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Memory Explorer</h1>
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

