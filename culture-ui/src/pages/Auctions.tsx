import { useEffect, useState } from 'react'
import { registerWidget } from '../lib/widgetRegistry'

interface Bid {
  agent_id: string
  amount: number
}

interface Auction {
  id: number
  item: string
  status: string
  winner_id: string | null
  bids: Bid[]
}

export default function Auctions() {
  const [data, setData] = useState<Auction[]>([])

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch('/api/auctions')
        const json = (await res.json()) as { auctions: Auction[] }
        if (!cancelled) setData(json.auctions || [])
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
      <h1 className="text-xl font-bold">Auctions</h1>
      <table className="min-w-full border" data-testid="auction-table">
        <thead>
          <tr>
            <th className="border px-2">ID</th>
            <th className="border px-2">Item</th>
            <th className="border px-2">Status</th>
            <th className="border px-2">Bids</th>
          </tr>
        </thead>
        <tbody>
          {data.map((a) => (
            <tr key={a.id}>
              <td className="border px-2">{a.id}</td>
              <td className="border px-2">{a.item}</td>
              <td className="border px-2">{a.status}</td>
              <td className="border px-2">
                {a.bids.map((b) => `${b.agent_id}:${b.amount}`).join(', ')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

registerWidget('Auctions', Auctions)
