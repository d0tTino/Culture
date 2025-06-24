import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import MissionOverview, { reorderMissions } from './pages/MissionOverview'
import * as api from './lib/api'

vi.mock('./lib/api')

const missions = [
  { id: 1, name: 'Gather Intel', status: 'In Progress', progress: 50 },
  { id: 2, name: 'Prepare Brief', status: 'Pending', progress: 0 },
  { id: 3, name: 'Execute Plan', status: 'Complete', progress: 100 },
]

;(api.fetchMissions as unknown as vi.Mock).mockResolvedValue(missions)

describe('MissionOverview', () => {
  it('renders missions table', async () => {
    render(<MissionOverview />)
    expect(await screen.findByRole('heading', { name: /mission overview/i })).toBeInTheDocument()
    expect(await screen.findByText('Gather Intel')).toBeInTheDocument()
    expect(screen.getByText('Prepare Brief')).toBeInTheDocument()
  })

  it('reorders rows helper', () => {
    const reordered = reorderMissions([...missions], missions[0].id, missions[1].id)
    expect(reordered[0].id).toBe(2)
    expect(reordered[1].id).toBe(1)
  })
})
