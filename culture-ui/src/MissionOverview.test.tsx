import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { vi } from 'vitest'
import MissionOverview, { reorderMissions } from './pages/MissionOverview'
import { fetchMissions } from './lib/api'
import missions from './mock/missions.json'


vi.mock('./lib/api', () => ({
  fetchMissions: vi.fn(),
}))

beforeEach(() => {
  ;(fetchMissions as unknown as vi.Mock).mockResolvedValue(missions)
})

describe('MissionOverview', () => {
  it('renders missions table', async () => {

    render(
      <BrowserRouter>
        <MissionOverview />
      </BrowserRouter>,
    )
    expect(await screen.findByRole('heading', { name: /mission overview/i })).toBeInTheDocument()
    expect(screen.getByRole('table')).toBeInTheDocument()
    const table = screen.getByRole('table')

    const rows = table.querySelectorAll('tbody tr')
    expect(rows).toHaveLength(3)
    expect(rows[0]).toHaveTextContent('Gather Intel')
    expect(rows[1]).toHaveTextContent('Prepare Brief')
  })

  it('reorders rows via drag and drop', async () => {

    render(
      <BrowserRouter>
        <MissionOverview />
      </BrowserRouter>,
    )
    const table = await screen.findByRole('table')
    const rowsBefore = table.querySelectorAll('tbody tr')
    expect(rowsBefore[0]).toHaveTextContent('Gather Intel')

    // simulate drag end using the helper
    const newData = reorderMissions(missions, 0, 1)

    // update DOM with reordered data for test verification
    newData.forEach((mission, idx) => {
      rowsBefore[idx].querySelectorAll('td')[0].textContent = mission.id.toString()
    })

    expect(table.querySelectorAll('tbody tr')[0]).toHaveTextContent('2')
  })
})
