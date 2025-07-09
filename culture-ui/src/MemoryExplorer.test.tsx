import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import App from './App'

vi.mock('./App.css', () => ({}))

describe('MemoryExplorer', () => {
  let fetchMock: ReturnType<typeof vi.fn>
  beforeEach(() => {
    fetchMock = vi.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ summaries: ['hello world'] }),
      }) as unknown as Response,
    )
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('fetches summaries for selected agent', async () => {
    render(
      <MemoryRouter initialEntries={["/memory"]}>
        <App />
      </MemoryRouter>,
    )

    expect(await screen.findByText('hello world')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/agents/agent-1/semantic_summaries',
    )

    fetchMock.mockResolvedValue({
      json: () => Promise.resolve({ summaries: ['second'] }),
    } as Response)

    const input = screen.getByLabelText('agent-select')
    await userEvent.clear(input)
    await userEvent.type(input, 'agent-2')

    expect(await screen.findByText('second')).toBeInTheDocument()
    expect(
      fetchMock.mock.calls.some((c) => c[0] === '/api/agents/agent-2/semantic_summaries'),
    ).toBe(true)
  })
})
