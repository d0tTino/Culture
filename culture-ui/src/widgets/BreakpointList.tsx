import { useState } from 'react'

const DEFAULT_TAGS = ['violence', 'nsfw', 'sabotage']

export default function BreakpointList() {
  const [tags] = useState<string[]>(DEFAULT_TAGS)

  return (
    <div className="p-2" data-testid="breakpoints">
      <h2 className="font-bold">Breakpoints</h2>
      <ul className="list-disc pl-4">
        {tags.map((tag) => (
          <li key={tag}>{tag}</li>
        ))}
      </ul>
    </div>
  )
}
