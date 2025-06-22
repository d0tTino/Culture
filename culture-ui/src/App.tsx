import { NavLink, Route, Routes } from 'react-router-dom'
import clsx from 'clsx'
import Home from './pages/Home'
import MissionOverview from './pages/MissionOverview'
import AgentDataOverview from './pages/AgentDataOverview'

export default function App() {
  return (
    <div className="flex min-h-screen">
      <aside className="w-48 border-r p-4">
        <nav className="space-y-2">
          <NavLink
            to="/"
            end
            className={({ isActive }) => clsx(isActive && 'font-bold text-brand')}
          >
            Home
          </NavLink>
          <br />
          <NavLink
            to="/missions"
            className={({ isActive }) => clsx(isActive && 'font-bold text-brand')}
          >
            Mission Overview
          </NavLink>
          <br />
          <NavLink
            to="/agent-data"
            className={({ isActive }) => clsx(isActive && 'font-bold text-brand')}
          >
            Agent Data Overview
          </NavLink>
        </nav>
      </aside>
      <main className="flex-1 p-4">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/missions" element={<MissionOverview />} />
          <Route path="/agent-data" element={<AgentDataOverview />} />
        </Routes>
      </main>
    </div>

  )
}
