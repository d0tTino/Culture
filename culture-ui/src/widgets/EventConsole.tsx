import { useEffect, useState } from 'react'
import { useEventSource } from '../lib/useEventSource'

interface AnyEvent {
  [key: string]: unknown
}

export default function EventConsole() {
  const event = useEventSource<AnyEvent>()
  const [events, setEvents] = useState<AnyEvent[]>([])
  const [search, setSearch] = useState('')

  useEffect(() => {
    if (event) {
      setEvents((cur) => [...cur, event])
    }
  }, [event])

  const filtered = events.filter((ev) =>
    JSON.stringify(ev).toLowerCase().includes(search.toLowerCase()),
  )

  return (
    <div className="p-2" data-testid="event-console">
      <input
        aria-label="search"
        placeholder="Search events"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="mb-2 border p-1"
      />
      <div data-testid="events" className="space-y-2 max-h-64 overflow-y-auto">
        {filtered.map((ev, i) => (
          <pre key={i} className="bg-muted p-2 text-xs">
            {JSON.stringify(ev, null, 2)}
          </pre>
        ))}
      </div>
    </div>
  )
}
