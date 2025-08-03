import { Routes, Route } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import HeaderBar from './components/HeaderBar'
import Home from './pages/Home'
import MissionOverview from './pages/MissionOverview'
import AgentDataOverview from './pages/AgentDataOverview'
import LiveMapPage from './pages/LiveMap'
import MapStatePage from './pages/MapState'
import AgentMemoriesPage from './pages/AgentMemories'
import NetworkWebPage from './pages/NetworkWeb'
import WorldMapPage from './pages/WorldMap'
import LawProposalPage from './pages/LawProposal'
import ProposalsPage from './pages/Proposals'
import TimelineWidgetPage from './pages/TimelineWidget'
import MemoryExplorerPage from './pages/MemoryExplorer'
import KpiCardPage from './pages/KpiCard'
import StoryboardPage from './pages/Storyboard'
import AgentTimelinePage from './pages/AgentTimeline'
import TokenBalancesPage from './pages/TokenBalances'
import AuctionsPage from './pages/Auctions'
import QuestsPage from './pages/Quests'
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
            <Route path="/live-map" element={<LiveMapPage />} />
            <Route path="/map-state" element={<MapStatePage />} />
            <Route path="/agent-memories" element={<AgentMemoriesPage />} />
            <Route path="/network-web" element={<NetworkWebPage />} />
            <Route path="/world-map" element={<WorldMapPage />} />
            <Route path="/timeline" element={<TimelineWidgetPage />} />
            <Route path="/agent-timeline" element={<AgentTimelinePage />} />
            <Route path="/propose-law" element={<LawProposalPage />} />
            <Route path="/proposals" element={<ProposalsPage />} />
            <Route path="/kpi-card" element={<KpiCardPage />} />
            <Route path="/storyboard" element={<StoryboardPage />} />
            <Route path="/balances" element={<TokenBalancesPage />} />
            <Route path="/auctions" element={<AuctionsPage />} />
            <Route path="/quests" element={<QuestsPage />} />
          </Routes>
        </DockManager>
      </main>

    </div>
  )
}
