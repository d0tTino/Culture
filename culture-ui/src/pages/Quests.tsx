import { useEffect, useState } from 'react'
import { registerWidget } from '../lib/widgetRegistry'
import { fetchQuests, type Quest } from '../lib/api'

export default function Quests() {
  const [quests, setQuests] = useState<Quest[]>([])
  const [fallback, setFallback] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const envelope = await fetchQuests()
        if (!cancelled) {
          setQuests(envelope.data.quests || [])
          setFallback(envelope.enabled ? null : envelope.fallback?.reason || 'Quests unavailable.')
        }
      } catch {
        if (!cancelled) {
          setFallback('Quests unavailable.')
        }
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Quests</h1>
      {fallback && <p data-testid="quests-fallback">{fallback}</p>}
      <table className="min-w-full border" data-testid="quest-table">
        <thead>
          <tr>
            <th className="border px-2">ID</th>
            <th className="border px-2">Title</th>
            <th className="border px-2">Status</th>
          </tr>
        </thead>
        <tbody>
          {quests.map((q) => (
            <tr key={q.id}>
              <td className="border px-2">{q.id}</td>
              <td className="border px-2">{q.title}</td>
              <td className="border px-2">{q.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

registerWidget('Quests', Quests)
