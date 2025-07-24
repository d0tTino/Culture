import { act, render, screen } from '@testing-library/react'
import LiveMapPage from './pages/LiveMap'
import { MockEventSource, resetMockSources } from './lib/testUtils'
import { vi } from 'vitest'

afterEach(() => {
  resetMockSources()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('LiveMap', () => {
  it('renders mood and summary', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve({
          json: () =>
            Promise.resolve({
              world_map: { agents: { 'agent-1': [0, 0] } },
              agents: { 'agent-1': { mood: 0.2, summary: 'hello' } },
            }),
        }) as unknown as Response,
      ),
    )
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource

    render(<LiveMapPage />)

    expect(await screen.findByText('agent-1: 0, 0 (mood 0.2) - hello')).toBeInTheDocument()

    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage('{"type":"map_change","data":{"world_map":{"agents":{"agent-1":[1,2]}}}}')
    })

    expect(await screen.findByText('agent-1: 1, 2 (mood 0.2) - hello')).toBeInTheDocument()
  })
})
