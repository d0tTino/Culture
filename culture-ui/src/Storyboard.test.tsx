import { act, render, screen, waitFor } from '@testing-library/react'
import StoryboardPage from './pages/Storyboard'
import { MockWebSocket, resetMockSources } from './lib/testUtils'
import { vi } from 'vitest'

afterEach(() => {
  resetMockSources()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('Storyboard widget', () => {
  it('shows coordinates, mood and summaries', async () => {
    ;(
      globalThis as unknown as { WebSocket?: typeof WebSocket }
    ).WebSocket = MockWebSocket as unknown as typeof WebSocket
    const fetchSpy = vi.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ summaries: ['s1'] }),
      }) as unknown as Response,
    )
    vi.stubGlobal('fetch', fetchSpy as unknown as typeof fetch)

    render(<StoryboardPage />)

    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.sendMessage(
        '{"data":{"world_map":{"agents":{"a1":[5,6]}},"agents":[{"agent_id":"a1","mood":0.2}]}}',
      )
    })

    expect(await screen.findByText('a1: 5, 6 (mood 0.2)')).toBeInTheDocument()

    act(() => {
      screen.getByRole('button', { name: /summaries/i }).click()
    })

    expect(await screen.findByText('s1')).toBeInTheDocument()
    expect(fetchSpy).toHaveBeenCalledWith('/api/agents/a1/semantic_summaries')

    act(() => {
      screen.getByRole('button', { name: /map/i }).click()
    })

    expect(screen.getByText('a1: 5, 6 (mood 0.2)')).toBeInTheDocument()
  })

  it('handles fetch errors gracefully', async () => {
    ;(
      globalThis as unknown as { WebSocket?: typeof WebSocket }
    ).WebSocket = MockWebSocket as unknown as typeof WebSocket
    const fetchSpy = vi.fn(() => Promise.reject(new Error('fail')))
    vi.stubGlobal('fetch', fetchSpy as unknown as typeof fetch)

    render(<StoryboardPage />)

    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.sendMessage(
        '{"data":{"world_map":{"agents":{"a1":[5,6]}}}}',
      )
    })

    expect(await screen.findByText('a1: 5, 6 (mood n/a)')).toBeInTheDocument()

    act(() => {
      screen.getByRole('button', { name: /summaries/i }).click()
    })

    await waitFor(() => expect(fetchSpy).toHaveBeenCalled())
    expect(screen.getByTestId('summaries').textContent).toBe('')

    act(() => {
      screen.getByRole('button', { name: /map/i }).click()
    })

    expect(screen.getByText('a1: 5, 6 (mood n/a)')).toBeInTheDocument()
  })

  it('renders heatmap and memory events', async () => {
    ;(
      globalThis as unknown as { WebSocket?: typeof WebSocket }
    ).WebSocket = MockWebSocket as unknown as typeof WebSocket

    render(<StoryboardPage />)

    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.sendMessage(
        '{"data":{"world_map":{"agents":{"a1":[0,0]}}}}',
      )
      ws.sendMessage(
        '{"event_type":"memory_prune","data":{"step":1}}',
      )
    })

    const heatmap = await screen.findByTestId('heatmap')
    const firstCell = heatmap.firstChild as HTMLElement
    await waitFor(() =>
      expect(firstCell.style.backgroundColor).toBe('rgba(255, 0, 0, 1)'),
    )

    expect(
      await screen.findByText(/memory_prune \(step 1\)/i),
    ).toBeInTheDocument()
  })
})

