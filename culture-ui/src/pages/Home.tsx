import { registerWidget } from '../lib/widgetRegistry'

export default function Home() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Welcome to Culture UI</h1>
      <p className="mt-2">Select a page from the sidebar.</p>
    </div>
  )
}

registerWidget('home', Home)
