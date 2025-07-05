import { act, render, screen } from '@testing-library/react'
import LiveMap from './pages/LiveMap'
import { MockEventSource, resetMockSources } from './lib/testUtils'
import { vi } from 'vitest'

afterEach(() => {
  resetMockSources()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('LiveMap', () => {
  it('renders positions and summaries from events', async () => {
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ summaries: ['summary1'] }),
      }) as unknown as Response,
    ))

    render(<LiveMap />)

    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage(
        '{"data":{"world_map":{"agents":{"agent-1":[1,2]}}}}',
      )
    })

    expect(await screen.findByText('agent-1: 1, 2')).toBeInTheDocument()
    expect(await screen.findByText('summary1')).toBeInTheDocument()
  })
})
