import { useEffect, useState } from 'react'
import { useEventSource } from '../lib/useEventSource'

interface BreakpointEvent {
  event_type?: string
  data?: { tags?: string[]; step?: number }
}

const DEFAULT_TAGS = ['violence', 'nsfw', 'sabotage']

export default function BreakpointList() {
  const [tags] = useState<string[]>(DEFAULT_TAGS)
  const [selected, setSelected] = useState<string[]>([])
  const [toast, setToast] = useState<string | null>(null)
  const event = useEventSource<BreakpointEvent>()

  useEffect(() => {
    if (event?.event_type === 'breakpoint_hit') {
      const hit = event.data?.tags?.join(', ') ?? ''
      setToast(`Breakpoint hit: ${hit}`)
      const t = setTimeout(() => setToast(null), 3000)
      return () => clearTimeout(t)
    }
  }, [event])

  const toggleTag = (tag: string) => {
    setSelected((cur) =>
      cur.includes(tag) ? cur.filter((t) => t !== tag) : [...cur, tag],
    )
  }

  const sendBreakpoints = async () => {
    await fetch('/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command: 'set_breakpoints', tags: selected }),
    })
  }

  return (
    <div className="p-2" data-testid="breakpoints">
      <h2 className="font-bold">Breakpoints</h2>
      <ul className="list-disc pl-4">
        {tags.map((tag) => (
          <li key={tag}>
            <label>
              <input
                type="checkbox"
                aria-label={tag}
                checked={selected.includes(tag)}
                onChange={() => toggleTag(tag)}
              />{' '}
              {tag}
            </label>
          </li>
        ))}
      </ul>
      <button onClick={sendBreakpoints}>Save</button>
      {toast && (
        <div data-testid="toast" role="alert">
          {toast}
        </div>
      )}
    </div>
  )
}
