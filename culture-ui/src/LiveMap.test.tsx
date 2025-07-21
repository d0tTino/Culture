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
  it('renders positions from events', async () => {
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource

    render(<LiveMapPage />)

    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage('{"type":"map_change","data":{"world_map":{"agents":{"agent-1":[1,2]}}}}')
    })

    expect(await screen.findByText('agent-1: 1, 2')).toBeInTheDocument()
  })
})
