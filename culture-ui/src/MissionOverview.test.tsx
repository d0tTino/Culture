import { render, screen } from '@testing-library/react'
import MissionOverview from './pages/MissionOverview'
import { reorderMissions } from './lib/reorderMissions'

const missions = [
  { id: 1, name: 'Gather Intel', status: 'In Progress', progress: 50 },
  { id: 2, name: 'Prepare Brief', status: 'Pending', progress: 0 },
  { id: 3, name: 'Execute Plan', status: 'Complete', progress: 100 },
]

describe('MissionOverview', () => {

  it('renders missions table', async () => {

    render(<MissionOverview />)
    expect(screen.getByRole('heading', { name: /mission overview/i })).toBeInTheDocument()
    expect(screen.getByText('Gather Intel')).toBeInTheDocument()
    expect(screen.getByText('Prepare Brief')).toBeInTheDocument()
  })

  it('reorders rows helper', () => {
    const reordered = reorderMissions([...missions], missions[0].id, missions[1].id)
    expect(reordered[0].id).toBe(2)
    expect(reordered[1].id).toBe(1)
  })
})
