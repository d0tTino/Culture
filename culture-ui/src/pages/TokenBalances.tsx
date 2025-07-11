import { useEffect, useState } from 'react'
import { registerWidget } from '../lib/widgetRegistry'

interface BalanceInfo {
  ip: number
  du: number
  tokens: Record<string, number>
}

export default function TokenBalances() {
  const [data, setData] = useState<Record<string, BalanceInfo>>({})

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch('/api/token_balances')
        const json = (await res.json()) as { agents: Record<string, BalanceInfo> }
        if (!cancelled) setData(json.agents || {})
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
      <h1 className="text-xl font-bold">Token Balances</h1>
      <table className="min-w-full border" data-testid="balance-table">
        <thead>
          <tr>
            <th className="border px-2">Agent</th>
            <th className="border px-2">IP</th>
            <th className="border px-2">DU</th>
            <th className="border px-2">Tokens</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data).map(([id, info]) => (
            <tr key={id}>
              <td className="border px-2">{id}</td>
              <td className="border px-2">{info.ip}</td>
              <td className="border px-2">{info.du}</td>
              <td className="border px-2">
                {Object.entries(info.tokens || {})
                  .map(([tok, amt]) => `${tok}:${amt}`)
                  .join(', ')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

registerWidget('TokenBalances', TokenBalances)
