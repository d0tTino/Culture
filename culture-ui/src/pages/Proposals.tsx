import { useEffect, useState } from 'react'
import { registerWidget } from '../lib/widgetRegistry'

interface Proposal {
  proposer_id: string
  text: string
  approved: boolean
  yes_weight: number
  no_weight: number
  ip_spent: number
  ts: string
}

export default function Proposals() {
  const [data, setData] = useState<Proposal[]>([])

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch('/api/recent_proposals')
        const json = (await res.json()) as { proposals: Proposal[] }
        if (!cancelled) setData(json.proposals || [])
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
      <h1 className="text-xl font-bold">Recent Proposals</h1>
      <table className="min-w-full border" data-testid="proposal-table">
        <thead>
          <tr>
            <th className="border px-2">Time</th>
            <th className="border px-2">Proposer</th>
            <th className="border px-2">Text</th>
            <th className="border px-2">Yes</th>
            <th className="border px-2">No</th>
            <th className="border px-2">Approved</th>
          </tr>
        </thead>
        <tbody>
          {data.map((p, idx) => (
            <tr key={idx}>
              <td className="border px-2">{new Date(p.ts).toLocaleString()}</td>
              <td className="border px-2">{p.proposer_id}</td>
              <td className="border px-2">{p.text}</td>
              <td className="border px-2">{p.yes_weight}</td>
              <td className="border px-2">{p.no_weight}</td>
              <td className="border px-2">{p.approved ? 'Yes' : 'No'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

registerWidget('Proposals', Proposals)
