import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import App from './App'

vi.mock('./App.css', () => ({}))

describe('MemoryExplorer', () => {
  let fetchMock: ReturnType<typeof vi.fn>
  beforeEach(() => {
    fetchMock = vi.fn((url: string) => {
      if (url === '/api/agents/agent-1/semantic_summaries') {
        return Promise.resolve({
          json: () => Promise.resolve({ summaries: ['hello world'] }),
        }) as unknown as Response
      }
      if (url === '/api/agents/agent-1/memories') {
        return Promise.resolve({
          json: () => Promise.resolve({ memories: [{ content: 'm1' }] }),
        }) as unknown as Response
      }
      if (url === '/api/agents/agent-2/semantic_summaries') {
        return Promise.resolve({
          json: () => Promise.resolve({ summaries: ['second'] }),
        }) as unknown as Response
      }
      if (url === '/api/agents/agent-2/memories') {
        return Promise.resolve({
          json: () => Promise.resolve({ memories: [{ content: 'm2' }] }),
        }) as unknown as Response
      }
      return Promise.resolve({ json: () => Promise.resolve({}) }) as Response
    })
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch)
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('fetches summaries for selected agent', async () => {
    render(
      <MemoryRouter initialEntries={["/memory"]}>
        <App />
      </MemoryRouter>,
    )

    expect(await screen.findByText('hello world')).toBeInTheDocument()
    expect(await screen.findByText('m1')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/agents/agent-1/semantic_summaries',
    )
    expect(fetchMock).toHaveBeenCalledWith('/api/agents/agent-1/memories')

    const input = screen.getByLabelText('agent-select')
    await userEvent.clear(input)
    await userEvent.type(input, 'agent-2')

    expect(await screen.findByText('second')).toBeInTheDocument()
    expect(await screen.findByText('m2')).toBeInTheDocument()
    expect(
      fetchMock.mock.calls.some(
        (c) => c[0] === '/api/agents/agent-2/semantic_summaries',
      ),
    ).toBe(true)
    expect(
      fetchMock.mock.calls.some((c) => c[0] === '/api/agents/agent-2/memories'),
    ).toBe(true)
  })

  it('keeps previous summaries when fetch fails', async () => {
    render(
      <MemoryRouter initialEntries={["/memory"]}>
        <App />
      </MemoryRouter>,
    )

    expect(await screen.findByText('hello world')).toBeInTheDocument()
    expect(await screen.findByText('m1')).toBeInTheDocument()

    fetchMock.mockImplementation(() => Promise.reject(new Error('fail')))

    const input = screen.getByLabelText('agent-select')
    await userEvent.clear(input)
    await userEvent.type(input, 'agent-2')

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/agents/agent-2/semantic_summaries',
      ),
    )

    expect(screen.queryByText('hello world')).toBeInTheDocument()
    expect(screen.queryByText('m1')).toBeInTheDocument()
    expect(screen.getByTestId('summaries').textContent?.trim()).toBe(
      'hello world',
    )
    expect(screen.getByTestId('memories').textContent?.trim()).toBe('m1')
  })
})
