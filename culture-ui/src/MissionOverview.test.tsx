import { vi } from 'vitest'
import type { Mission } from './lib/api'

var missions: Mission[] = []

vi.mock('./lib/api', () => {
  missions = [
    { id: 1, name: 'Gather Intel', status: 'In Progress', progress: 50 },
    { id: 2, name: 'Prepare Brief', status: 'Pending', progress: 0 },
    { id: 3, name: 'Execute Plan', status: 'Complete', progress: 100 },
  ]
  return {
    fetchMissions: vi.fn().mockResolvedValue(missions),
  }
})

import { render, screen } from '@testing-library/react'
import MissionOverview, { reorderMissions } from './pages/MissionOverview'

describe('MissionOverview', () => {
  it('renders missions table', async () => {
    render(<MissionOverview />)
    expect(await screen.findByRole('heading', { name: /mission overview/i })).toBeInTheDocument()
    const table = await screen.findByRole('table')

    const rows = table.querySelectorAll('tbody tr')
    expect(rows).toHaveLength(3)
    expect(rows[0]).toHaveTextContent('Gather Intel')
    expect(rows[1]).toHaveTextContent('Prepare Brief')
  })

})
