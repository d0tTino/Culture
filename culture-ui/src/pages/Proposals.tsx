import { useEffect, useState } from 'react'
import { registerWidget } from '../lib/widgetRegistry'
import { submitProposal } from '../lib/api'

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
  const [text, setText] = useState('')
  const [proposer, setProposer] = useState('agent_1')
  const [result, setResult] = useState<string | null>(null)

  async function load() {
    try {
      const res = await fetch('/api/recent_proposals')
      const json = (await res.json()) as { proposals: Proposal[] }
      setData(json.proposals || [])
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    void load()
  }, [])

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setResult(null)
    try {
      const outcome = await submitProposal(proposer, text)
      setResult(outcome.approved ? 'Approved' : 'Rejected')
      await load()
    } catch {
      setResult('Error')
    }
  }

  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Recent Proposals</h1>
      <form onSubmit={submit} className="space-y-2">
        <input
          placeholder="Proposal text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="border p-1 w-full"
        />
        <input
          placeholder="Proposer ID"
          value={proposer}
          onChange={(e) => setProposer(e.target.value)}
          className="border p-1 w-full"
        />
        <button type="submit" className="px-2 py-1 border rounded">
          Submit
        </button>
        {result && <div data-testid="result">Result: {result}</div>}
      </form>
      <table className="min-w-full border" data-testid="proposal-table">
        <thead>
          <tr>
            <th className="border px-2">Time</th>
            <th className="border px-2">Proposer</th>
            <th className="border px-2">Text</th>
            <th className="border px-2">Yes</th>
            <th className="border px-2">No</th>
            <th className="border px-2">IP Spent</th>
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
              <td className="border px-2">{p.ip_spent}</td>
              <td className="border px-2">{p.approved ? 'Yes' : 'No'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

registerWidget('Proposals', Proposals)
