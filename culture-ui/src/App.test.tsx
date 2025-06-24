import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import App from './App'

vi.mock('./App.css', () => ({}))

describe('App', () => {
  it('renders home page', () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
      </MemoryRouter>,
    )

    expect(
      screen.getByRole('heading', { name: /welcome to culture ui/i }),
    ).toBeInTheDocument()
  })
})
