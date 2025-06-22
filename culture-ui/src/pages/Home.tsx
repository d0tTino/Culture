import { widgetRegistry } from '../lib/widgetRegistry'

export default function Home() {
  const Timeline = widgetRegistry.get('Timeline')
  const Breakpoints = widgetRegistry.get('Breakpoints')
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Welcome to Culture UI</h1>
      <p className="mt-2">Select a page from the sidebar.</p>
      {Timeline && <Timeline />}
      {Breakpoints && <Breakpoints />}
    </div>
  )
}
