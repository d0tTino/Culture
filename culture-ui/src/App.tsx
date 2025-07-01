import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import Home from './pages/Home'
import MissionOverview from './pages/MissionOverview'
import AgentDataOverview from './pages/AgentDataOverview'
import DockManager from './components/DockManager'
import { createDefaultLayout } from './lib/defaultLayout'


export default function App() {
  const defaultLayout = createDefaultLayout()
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-y-auto">
        <Header />
        <main className="flex-1 overflow-y-auto">
          <DockManager defaultLayout={defaultLayout}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/missions" element={<MissionOverview />} />
              <Route path="/agent-data" element={<AgentDataOverview />} />
            </Routes>
          </DockManager>
        </main>
      </div>
    </div>
  )
}
