import { act, render, screen } from '@testing-library/react'
import StoryboardPage from './pages/Storyboard'
import { MockEventSource, resetMockSources } from './lib/testUtils'
import { vi } from 'vitest'

afterEach(() => {
  resetMockSources()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('Storyboard widget', () => {
  it('shows coordinates, mood and summaries', async () => {
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ summaries: ['s1'] }),
      }) as unknown as Response,
    ))

    render(<StoryboardPage />)

    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage(
        '{"data":{"world_map":{"agents":{"a1":[5,6]}},"agents":[{"agent_id":"a1","mood":0.2}]}}',
      )
    })

    expect(await screen.findByText('a1: 5, 6 (mood 0.2)')).toBeInTheDocument()

    act(() => {
      screen.getByRole('button', { name: /summaries/i }).click()
    })

    expect(await screen.findByText('s1')).toBeInTheDocument()
  })
})

