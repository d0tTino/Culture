import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import Quests from './pages/Quests'

describe('Quests page', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('renders quests from versioned api envelope', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string) =>
      Promise.resolve({
        json: () => Promise.resolve(
          url === '/api/quests'
            ? { schema: 'dashboard.quests', version: '2026-03-18', enabled: true, data: { quests: [{ id: 1, title: 'Q1', description: '', progress: 0, status: 'pending' }] } }
            : { schema: 'dashboard.capabilities', version: '2026-03-18', enabled: true, data: { capabilities: {} } },
        )
      }) as unknown as Response
    ))
    render(<Quests />)
    expect(await screen.findByText('Q1')).toBeInTheDocument()
  })
})
