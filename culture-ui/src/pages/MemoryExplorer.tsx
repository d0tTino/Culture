import { useEffect, useState } from 'react'
import { registerWidget } from '../lib/widgetRegistry'

interface SummaryResponse {
  summaries: string[]
}

export default function MemoryExplorerPage() {
  const [summaries, setSummaries] = useState<string[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/agents/agent-1/semantic_summaries')
      .then((res) => res.json() as Promise<SummaryResponse>)
      .then((data) => setSummaries(data.summaries ?? []))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="p-4 space-y-4" data-testid="memory-explorer">
      <h1 className="text-xl font-bold">Memory Explorer</h1>
      {loading ? (
        <div>Loading...</div>
      ) : (
        <ul className="space-y-2 max-h-64 overflow-y-auto">
          {summaries.map((s, i) => (
            <li key={i} className="border p-2 rounded">
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

registerWidget('MemoryExplorer', MemoryExplorerPage)
