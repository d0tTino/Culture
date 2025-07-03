import EventConsole from '../widgets/EventConsole'
import BreakpointList from '../widgets/BreakpointList'

export default function MemoryExplorer() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Memory Explorer</h1>
      <div className="grid gap-4 grid-cols-2">
        <BreakpointList />
        <EventConsole />
      </div>
    </div>
  )
}

