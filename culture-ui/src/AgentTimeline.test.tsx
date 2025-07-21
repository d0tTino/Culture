import { act, render, screen } from '@testing-library/react'
import AgentTimelinePage from './pages/AgentTimeline'
import { MockEventSource, resetMockSources } from './lib/testUtils'
import { vi } from 'vitest'

afterEach(() => {
  resetMockSources()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('AgentTimeline widget', () => {
  it('shows positions and summaries from events', async () => {
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ summaries: ['sum1'] }),
      }) as unknown as Response,
    ))

    render(<AgentTimelinePage />)

    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage(
        '{"type":"update","data":{"step":1,"world_map":{"agents":{"agent-1":[3,4]}}}}',
      )
    })

    expect(await screen.findByText('agent-1: 3, 4')).toBeInTheDocument()
    expect(await screen.findByText('sum1')).toBeInTheDocument()
    expect(screen.getByRole('slider')).toHaveAttribute('max', '1')
  })
})
