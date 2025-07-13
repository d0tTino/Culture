import { act, render, screen } from '@testing-library/react'
import WorldMap from './pages/WorldMap'
import { MockEventSource, resetMockSources } from './lib/testUtils'
import { vi } from 'vitest'

afterEach(() => {
  resetMockSources()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('WorldMapPage', () => {
  it('renders positions and resources from events', async () => {
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource
    render(<WorldMap />)
    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage(
        '{"data":{"world_map":{"agents":{"agent-1":[1,2]},"resources":{"[0,0]":{"wood":1}}}}}'
      )
    })
    expect(await screen.findByText('agent-1: 1, 2')).toBeInTheDocument()
    expect(await screen.findByText('[0,0]: wood:1')).toBeInTheDocument()
  })
})
