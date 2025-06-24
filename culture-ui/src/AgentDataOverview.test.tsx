import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import App from './App'
import { vi } from 'vitest'

vi.mock('./App.css', () => ({}))

describe('AgentDataOverview widget', () => {
  it('renders Agent Data Overview component', () => {
    render(
      <MemoryRouter initialEntries={["/agent-data"]}>
        <App />
      </MemoryRouter>,
    )

    expect(
      screen.getByRole('heading', { name: /agent data overview/i }),
    ).toBeInTheDocument()

  })
})
