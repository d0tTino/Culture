import { useEffect, useMemo, useState } from 'react'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts'

interface Transaction {
  delta_ip: number
  delta_du: number
  reason: string
  gas_price_per_call: number
  gas_price_per_token: number
  ts: string
}

interface Props {
  agentId: string
  limit?: number
}

export default function ResourceHistory({ agentId, limit = 10 }: Props) {
  const [txns, setTxns] = useState<Transaction[]>([])

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch(`/api/transactions/${agentId}?limit=${limit}`)
        const json = (await res.json()) as Transaction[]
        if (!cancelled) setTxns(json || [])
      } catch {
        /* ignore */
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [agentId, limit])

  const data = useMemo(() => {
    let ip = 0
    let du = 0
    return [...txns].reverse().map((t, i) => {
      ip += t.delta_ip
      du += t.delta_du
      return { index: i, ip, du }
    })
  }, [txns])

  return (
    <div data-testid="resource-history" style={{ width: 400, height: 300 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <XAxis dataKey="index" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="ip" stroke="#8884d8" />
          <Line type="monotone" dataKey="du" stroke="#82ca9d" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
