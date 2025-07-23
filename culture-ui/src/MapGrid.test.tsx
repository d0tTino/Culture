import { act, render, screen } from '@testing-library/react'
import MapGrid from './widgets/MapGrid'
import { MockEventSource, resetMockSources } from './lib/testUtils'
import { vi } from 'vitest'

afterEach(() => {
  resetMockSources()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('MapGrid', () => {
  it('renders initial map and updates from events', async () => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ world_map: { width: 2, height: 2, agents: { a1: [0, 1] } } })
      }) as unknown as Response
    ))
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource

    render(<MapGrid />)

    expect(await screen.findByTestId('agent-a1')).toBeInTheDocument()

    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage('{"type":"map_change","data":{"world_map":{"width":2,"height":2,"agents":{"a1":[1,1]}}}}')
    })

    expect(await screen.findByTestId('agent-a1')).toBeInTheDocument()
  })
})
