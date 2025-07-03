import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import App from './App'

vi.mock('./App.css', () => ({}))

describe('MemoryExplorer', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ summaries: ['hello world'] }),
      }) as unknown as Response,
    ))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders semantic summaries', async () => {
    render(
      <MemoryRouter initialEntries={["/memory"]}>
        <App />
      </MemoryRouter>,
    )

    expect(await screen.findByText('hello world')).toBeInTheDocument()
  })
})
