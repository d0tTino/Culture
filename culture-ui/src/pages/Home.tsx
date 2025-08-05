import { registerWidget } from '../lib/widgetRegistry'
import AgentPositions from '../components/AgentPositions'
import AgentEmotions from '../components/AgentEmotions'
import AgentMemories from '../components/AgentMemories'

export default function Home() {
  return (
    <div className="p-4 space-y-4">
      <h1 className="text-xl font-bold">Welcome to Culture UI</h1>
      <p className="mt-2">Select a page from the sidebar.</p>
      <AgentPositions />
      <AgentEmotions />
      <AgentMemories />
    </div>
  )
}

registerWidget('home', Home)
