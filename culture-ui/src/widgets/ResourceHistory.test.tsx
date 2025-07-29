import { render, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import ResourceHistory from './ResourceHistory'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('ResourceHistory widget', () => {
  it('renders chart from transaction data', async () => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve([
          { delta_ip: 1, delta_du: 0, reason: 'r', gas_price_per_call: 0, gas_price_per_token: 0, ts: 't1' },
          { delta_ip: -0.5, delta_du: 0, reason: 's', gas_price_per_call: 0, gas_price_per_token: 0, ts: 't2' },
        ])
      }) as unknown as Response
    ))

    render(<ResourceHistory agentId="agent-1" />)

    expect(await screen.findByTestId('resource-history')).toBeInTheDocument()
  })
})
