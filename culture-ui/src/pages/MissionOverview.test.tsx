import { vi, type Mock } from 'vitest'

vi.mock('../lib/api', () => ({
  fetchMissions: vi.fn(),
}))

import { render, screen } from '@testing-library/react'
import MissionOverview from './MissionOverview'
import { fetchMissions } from '../lib/api'

const missions = [
  { id: 1, name: 'Test Mission', status: 'Pending', progress: 0 },
]

describe('MissionOverview', () => {
  it('renders missions returned by fetchMissions', async () => {
    ;(fetchMissions as unknown as Mock).mockResolvedValue(missions)
    render(<MissionOverview />)
    expect(await screen.findByText('Test Mission')).toBeInTheDocument()
  })
})
