import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import Quests from './pages/Quests'

describe('Quests page', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('renders quests from api', async () => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ quests: [{ id: 1, title: 'Q1', description: '', progress: 0, status: 'pending' }] })
      }) as unknown as Response
    ))
    render(<Quests />)
    expect(await screen.findByText('Q1')).toBeInTheDocument()
  })
})
