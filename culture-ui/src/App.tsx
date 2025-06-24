import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Home from './pages/Home'
import MissionOverview from './pages/MissionOverview'
import AgentDataOverview from './pages/AgentDataOverview'
import { registerWidget } from './lib/widgetRegistry'

registerWidget('home', Home)
registerWidget('missions', MissionOverview)
registerWidget('agentData', AgentDataOverview)

export default function App() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/missions" element={<MissionOverview />} />
          <Route path="/agent-data" element={<AgentDataOverview />} />
        </Routes>
      </main>
    </div>
  )
}
