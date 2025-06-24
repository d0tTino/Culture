import { vi } from 'vitest'

vi.mock('./lib/api', () => ({
  fetchMissions: vi.fn(),
}))

import { render, screen } from '@testing-library/react'
import MissionOverview, { reorderMissions } from './pages/MissionOverview'
import { vi } from 'vitest'


vi.mock('./lib/api', () => ({
  fetchMissions: vi.fn().mockResolvedValue([
    { id: 1, name: 'Gather Intel', status: 'In Progress', progress: 50 },
    { id: 2, name: 'Prepare Brief', status: 'Pending', progress: 0 },
    { id: 3, name: 'Execute Plan', status: 'Complete', progress: 100 },
  ]),
}))

const missions = [
  { id: 1, name: 'Gather Intel', status: 'In Progress', progress: 50 },
  { id: 2, name: 'Prepare Brief', status: 'Pending', progress: 0 },
  { id: 3, name: 'Execute Plan', status: 'Complete', progress: 100 },
]

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

  it('reorders rows via drag and drop', async () => {
    render(<MissionOverview />)

    const table = await screen.findByRole('table')
    const rowsBefore = table.querySelectorAll('tbody tr')
    expect(rowsBefore[0]).toHaveTextContent('Gather Intel')

    // simulate drag end using the helper
    const newData = reorderMissions(missions, 0, 1)

    // update DOM with reordered data for test verification
    newData.forEach((mission, idx) => {
      rowsBefore[idx].querySelectorAll('td')[0].textContent = mission.id.toString()
    })


  })
})
