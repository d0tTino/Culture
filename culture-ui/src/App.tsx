import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import MissionOverview from './pages/MissionOverview'
import AgentDataOverview from './pages/AgentDataOverview'
import './App.css'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen p-8">
        <nav className="mb-4 space-x-4">
          <Link to="/">Missions</Link>
          <Link to="/agent-data">Agent Data</Link>
        </nav>
        <Routes>
          <Route path="/" element={<MissionOverview />} />
          <Route path="/agent-data" element={<AgentDataOverview />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
