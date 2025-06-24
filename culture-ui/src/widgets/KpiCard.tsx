import { useEffect, useState } from 'react'
import { useEventSource } from '../lib/useEventSource'

interface KpiEvent {
  data?: { value?: number }
}

export default function KpiCard() {
  const event = useEventSource<KpiEvent>()
  const [values, setValues] = useState<number[]>([])

  useEffect(() => {
    const val = event?.data?.value
    if (typeof val === 'number') {
      setValues((v) => [...v, val])
    }
  }, [event])

  const latest = values.length ? values[values.length - 1] : 0

  return (
    <div data-testid="kpi-card">
      <div data-testid="kpi-value">{latest}</div>
      <div data-testid="chart">
        {values.map((v, i) => (
          <span key={i}>{v}</span>
        ))}
      </div>
    </div>
  )
}

