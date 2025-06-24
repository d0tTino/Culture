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
      </ul>
    </nav>
  )
}
