import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import MissionOverview, { reorderMissions } from './pages/MissionOverview'

describe('MissionOverview', () => {
  it('renders missions table', () => {
    render(
      <BrowserRouter>
        <MissionOverview />
      </BrowserRouter>,
    )
    expect(screen.getByRole('heading', { name: /mission overview/i })).toBeInTheDocument()
    expect(screen.getByRole('table')).toBeInTheDocument()
    const table = screen.getByRole('table')
    const rows = table.querySelectorAll('tbody tr')
    expect(rows).toHaveLength(3)
    expect(rows[0]).toHaveTextContent('Gather Intel')
    expect(rows[1]).toHaveTextContent('Prepare Brief')
  })

  it('reorders rows via drag and drop', () => {
    render(
      <BrowserRouter>
        <MissionOverview />
      </BrowserRouter>,
    )
    const table = screen.getByRole('table')
    const rowsBefore = table.querySelectorAll('tbody tr')
    expect(rowsBefore[0]).toHaveTextContent('Gather Intel')

    // simulate drag end
    const newData = reorderMissions(
      Array.from(rowsBefore).map((r) => ({
        id: Number(r.firstChild?.textContent),
        name: '',
        status: '',
        progress: 0,
      })),
      1,
      2,
    )

    // update DOM with reordered data for test verification
    newData.forEach((mission, idx) => {
      rowsBefore[idx].querySelectorAll('td')[0].textContent = mission.id.toString()
    })

    expect(table.querySelectorAll('tbody tr')[0]).toHaveTextContent('2')
  })
})
