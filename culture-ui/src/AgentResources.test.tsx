import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import AgentResources from './pages/AgentResources'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('AgentResources', () => {
  it('displays resources from API', async () => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        json: () =>
          Promise.resolve({
            agents: { 'agent-1': { ip: 1, du: 2, tokens: { TOK: 3 } } },
          }),
      }) as unknown as Response,
    ))

    render(<AgentResources />)

    expect(await screen.findByText('agent-1')).toBeInTheDocument()
    expect(screen.getByText('1')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
    expect(screen.getByText('TOK:3')).toBeInTheDocument()
  })
})
