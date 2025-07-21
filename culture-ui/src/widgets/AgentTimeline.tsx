import { useEffect, useState } from 'react'
import { useEventSource } from '../lib/useEventSource'

interface SimEvent {
  type?: string
  data?: {
    step?: number
    world_map?: { agents?: Record<string, [number, number]> }
  }
}

interface StepData {
  positions: Record<string, [number, number]>
}

export default function AgentTimeline() {
  const event = useEventSource<SimEvent>()
  const [history, setHistory] = useState<Record<number, StepData>>({})
  const [latestStep, setLatestStep] = useState(0)
  const [scrubStep, setScrubStep] = useState(0)
  const [summaries, setSummaries] = useState<string[]>([])

  useEffect(() => {
    const step = event?.data?.step
    if (typeof step === 'number') {
      setLatestStep(step)
      if (event?.data?.world_map?.agents) {
        setHistory((cur) => ({
          ...cur,
          [step]: { positions: event.data!.world_map!.agents! },
        }))
      }
    }
  }, [event])

  const positions =
    history[scrubStep]?.positions || history[latestStep]?.positions || {}

  useEffect(() => {
    const agentId = Object.keys(positions)[0]
    if (!agentId) {
      setSummaries([])
      return
    }
    let cancelled = false
    async function load() {
      try {
        const res = await fetch(`/api/agents/${agentId}/semantic_summaries`)
        const json = await res.json()
        if (!cancelled) setSummaries(json.summaries || [])
      } catch {
        /* ignore */
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [positions])

  return (
    <div className="p-2" data-testid="agent-timeline">
      <label htmlFor="agent-timeline-slider" className="block text-sm font-bold">
        Step {scrubStep || latestStep}
      </label>
      <input
        id="agent-timeline-slider"
        type="range"
        min={0}
        max={latestStep}
        value={scrubStep}
        onChange={(e) => setScrubStep(Number(e.target.value))}
      />
      <div data-testid="positions" className="mt-2">
        <ul>
          {Object.entries(positions).map(([id, pos]) => (
            <li key={id}>
              {id}: {pos[0]}, {pos[1]}
            </li>
          ))}
        </ul>
      </div>
      <div data-testid="summaries" className="mt-2">
        {summaries.map((s, i) => (
          <div key={i}>{s}</div>
        ))}
      </div>
    </div>
  )
}

