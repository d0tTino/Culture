import { NavLink } from 'react-router-dom'

export default function Sidebar() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    isActive ? 'font-bold' : undefined

  return (
    <nav className="p-4 w-48 bg-gray-100">
      <ul className="space-y-2">
        <li>
          <NavLink end to="/" className={linkClass}>
            Home
          </NavLink>
        </li>
        <li>
          <NavLink to="/missions" className={linkClass}>
            Mission Overview
          </NavLink>
        </li>
        <li>
          <NavLink to="/agent-data" className={linkClass}>
            Agent Data
          </NavLink>
        </li>
        <li>
          <NavLink to="/network-web" className={linkClass}>
            Network Web
          </NavLink>
        </li>
        <li>
          <NavLink to="/world-map" className={linkClass}>
            World Map
          </NavLink>
        </li>
        <li>
          <NavLink to="/timeline" className={linkClass}>
            Timeline Widget
          </NavLink>
        </li>
        <li>
          <NavLink to="/memory" className={linkClass}>
            Memory Explorer
          </NavLink>
        </li>
        <li>
          <NavLink to="/kpi-card" className={linkClass}>
            KPI Card
          </NavLink>
        </li>
        <li>
          <NavLink to="/memory" className={linkClass}>
            Memory Explorer
          </NavLink>
        </li>
      </ul>
    </nav>
  )
}
