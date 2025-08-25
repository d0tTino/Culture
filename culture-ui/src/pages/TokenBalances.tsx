import { useEffect, useState } from 'react'
import { fetchObservabilityMetrics } from '../lib/api'
import { registerWidget } from '../lib/widgetRegistry'

interface BalanceInfo {
  ip: number
  du: number
  tokens: Record<string, number>
}

export default function TokenBalances() {
  const [data, setData] = useState<Record<string, BalanceInfo>>({})
  const [metrics, setMetrics] = useState({
    agent_du_per_1k_tokens: {} as Record<string, number>,
    agent_llm_latency_p95_ms: {} as Record<string, number>,
  })

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
    async function loadMetrics() {
      try {
        const m = await fetchObservabilityMetrics()
        if (!cancelled)
          setMetrics({
            agent_du_per_1k_tokens: m.agent_du_per_1k_tokens || {},
            agent_llm_latency_p95_ms: m.agent_llm_latency_p95_ms || {},
          })
      } catch {
        /* ignore */
      }
    }
    void load()
    void loadMetrics()
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
            <th className="border px-2">DU/1k</th>
            <th className="border px-2">Latency</th>
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
              <td className="border px-2">
                {metrics.agent_du_per_1k_tokens[id] ?? 'n/a'}
              </td>
              <td className="border px-2">
                {metrics.agent_llm_latency_p95_ms[id] ?? 'n/a'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

registerWidget('TokenBalances', TokenBalances)
