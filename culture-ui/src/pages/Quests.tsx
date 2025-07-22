import { useEffect, useState } from 'react'
import { registerWidget } from '../lib/widgetRegistry'

interface Quest {
  id: number
  title: string
  description: string
  progress: number
  status: string
}

export default function Quests() {
  const [quests, setQuests] = useState<Quest[]>([])

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch('/api/quests')
        const json = (await res.json()) as { quests: Quest[] }
        if (!cancelled) setQuests(json.quests || [])
      } catch {
        /* ignore */
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
