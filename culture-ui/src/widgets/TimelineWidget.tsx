import { useState } from 'react'
import { useEventSource } from '../lib/useEventSource'

interface SimEvent {
  event_type: string
  data?: { step?: number }
}

export default function TimelineWidget() {
  const event = useEventSource<SimEvent>()
  const latestStep = event?.data?.step ?? 0
  const [scrubStep, setScrubStep] = useState(0)

  return (
    <div className="p-2" data-testid="timeline">
      <label htmlFor="timeline-slider" className="block text-sm font-bold">
        Step {scrubStep || latestStep}
      </label>
      <input
        id="timeline-slider"
        type="range"
        min={0}
        max={latestStep}
        value={scrubStep}
        onChange={(e) => setScrubStep(Number(e.target.value))}
      />
    </div>
  )
}
