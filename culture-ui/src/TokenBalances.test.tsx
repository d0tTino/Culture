import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import TokenBalances from './pages/TokenBalances'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('TokenBalances', () => {
  it('shows DU/1k and latency metrics', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo) => {
        if (input === '/api/token_balances') {
          return Promise.resolve({
            json: () =>
              Promise.resolve({
                agents: { 'agent-1': { ip: 1, du: 2, tokens: { TOK: 3 } } },
              }),
          }) as unknown as Response
        }
        return Promise.resolve({
          json: () =>
            Promise.resolve({
              agent_du_per_1k_tokens: { 'agent-1': 4.5 },
              agent_llm_latency_p95_ms: { 'agent-1': 123 },
            }),
        }) as unknown as Response
      }) as typeof fetch,
    )

    render(<TokenBalances />)

    expect(await screen.findByText('agent-1')).toBeInTheDocument()
    expect(screen.getByText('4.5')).toBeInTheDocument()
    expect(screen.getByText('123')).toBeInTheDocument()
  })
})

