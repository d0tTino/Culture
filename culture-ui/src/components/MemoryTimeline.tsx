
export interface MemoryEvent {
  type: string
  step?: number
}

export interface MemoryTimelineProps {
  events: MemoryEvent[]
}

export default function MemoryTimeline({ events }: MemoryTimelineProps) {
  return (
    <div
      data-testid="memory-events"
      className="max-h-32 overflow-y-auto mt-2 text-sm"
    >
      {events.map((e, i) => (
        <div key={i}>
          {e.type}
          {e.step !== undefined ? ` (step ${e.step})` : ''}
        </div>
      ))}
    </div>
  )
}
