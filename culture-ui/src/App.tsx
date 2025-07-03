import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import HeaderBar from './components/HeaderBar'
import Home from './pages/Home'
import MissionOverview from './pages/MissionOverview'
import AgentDataOverview from './pages/AgentDataOverview'
import MemoryExplorerPage from './pages/MemoryExplorer'
import NetworkWebPage from './pages/NetworkWeb'
import WorldMapPage from './pages/WorldMap'
import TimelineWidgetPage from './pages/TimelineWidget'
import KpiCardPage from './pages/KpiCard'
import DockManager from './components/DockManager'
import { createDefaultLayout } from './lib/defaultLayout'


export default function App() {
  const defaultLayout = createDefaultLayout()
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <HeaderBar />
        <DockManager defaultLayout={defaultLayout}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/missions" element={<MissionOverview />} />
            <Route path="/agent-data" element={<AgentDataOverview />} />
            <Route path="/memory" element={<MemoryExplorerPage />} />
            <Route path="/network-web" element={<NetworkWebPage />} />
            <Route path="/world-map" element={<WorldMapPage />} />
            <Route path="/timeline" element={<TimelineWidgetPage />} />
            <Route path="/kpi-card" element={<KpiCardPage />} />
          </Routes>
        </DockManager>
      </main>

    </div>
  )
}
