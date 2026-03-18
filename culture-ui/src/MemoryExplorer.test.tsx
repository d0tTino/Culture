import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import App from './App'
import { MockEventSource, resetMockSources } from './lib/testUtils'

vi.mock('./App.css', () => ({}))

describe('MemoryExplorer', () => {
  let fetchMock: ReturnType<typeof vi.fn>
  beforeEach(() => {
    resetMockSources()
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource
    fetchMock = vi.fn((url: string) => {
      if (url === '/api/capabilities') {
        return Promise.resolve({
          json: () => Promise.resolve({ schema: 'dashboard.capabilities', version: '2026-03-18', enabled: true, data: { capabilities: { memory: { enabled: true, reason: 'ok' } } } }),
        }) as unknown as Response
      }
      if (url === '/api/map') {
        return Promise.resolve({
          json: () => Promise.resolve({ schema: 'dashboard.map_state', version: '2026-03-18', enabled: true, data: { agents: { 'agent-1': {}, 'agent-2': {} } } }),
        }) as unknown as Response
      }
      if (url === '/api/memory/agent-1') {
        return Promise.resolve({
          json: () => Promise.resolve({ schema: 'dashboard.memory_views', version: '2026-03-18', enabled: true, data: { semantic: ['s1'], episodic: ['e1'] } }),
        }) as unknown as Response
      }
      if (url === '/api/memory/agent-2') {
        return Promise.resolve({
          json: () => Promise.resolve({ schema: 'dashboard.memory_views', version: '2026-03-18', enabled: true, data: { semantic: ['s2'], episodic: ['e2'] } }),
        }) as unknown as Response
      }
      return Promise.resolve({ json: () => Promise.resolve({}) }) as Response
    })
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch)
  })

  afterEach(() => {
    resetMockSources()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('loads map, memories, and streams messages', async () => {
    render(
      <MemoryRouter initialEntries={["/memory"]}>
        <App />
      </MemoryRouter>,
    )

    expect(fetchMock).toHaveBeenCalledWith('/api/map')
    expect(await screen.findByText('s1')).toBeInTheDocument()
    expect(await screen.findByText('e1')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith('/api/memory/agent-1')

    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage('{"agent_id":"agent-1","content":"hello"}')
    })
    expect(await screen.findByText('hello')).toBeInTheDocument()

    const select = screen.getByLabelText('agent-select')
    await userEvent.selectOptions(select, 'agent-2')
    expect(fetchMock).toHaveBeenCalledWith('/api/memory/agent-2')
    expect(await screen.findByText('s2')).toBeInTheDocument()
    expect(await screen.findByText('e2')).toBeInTheDocument()
  })
})
