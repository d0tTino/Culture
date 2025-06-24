import { render, screen } from '@testing-library/react'
import MissionOverview from './pages/MissionOverview'
import { reorderMissions } from './lib/reorderMissions'
import missions from './mock/missions.json'

;(api.fetchMissions as unknown as vi.Mock).mockResolvedValue(missions)

describe('MissionOverview', () => {
  it('renders missions table', () => {
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
