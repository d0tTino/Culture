import { vi, type MockInstance } from 'vitest'

vi.mock('./lib/api', () => ({
  fetchMissions: vi.fn(),
}))

import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import MissionOverview from './pages/MissionOverview'
import { reorderMissions } from './lib/reorderMissions'
import { fetchMissions } from './lib/api'

const missions = [
  { id: 1, name: 'Gather Intel', status: 'In Progress', progress: 50 },
  { id: 2, name: 'Prepare Brief', status: 'Pending', progress: 0 },
  { id: 3, name: 'Execute Plan', status: 'Complete', progress: 100 },
]

beforeEach(() => {
  ;(fetchMissions as unknown as MockInstance).mockResolvedValue(missions)
})

describe('MissionOverview', () => {
  it('renders missions table', async () => {
    render(
      <BrowserRouter>
        <MissionOverview />
      </BrowserRouter>,
    )
    expect(await screen.findByRole('heading', { name: /mission overview/i })).toBeInTheDocument()
    const table = await screen.findByRole('table')
    const rows = table.querySelectorAll('tbody tr')
    expect(rows).toHaveLength(3)
    expect(rows[0]).toHaveTextContent('Gather Intel')
    expect(rows[1]).toHaveTextContent('Prepare Brief')
  })

  it('reorders rows via drag and drop', () => {
    const newData = reorderMissions(missions, 1, 2)
    expect(newData[0].id).toBe(2)
    expect(newData[1].id).toBe(1)
    expect(newData[2].id).toBe(3)
  })
})
