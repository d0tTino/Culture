import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { vi } from 'vitest'
import App from './App'

vi.mock('./App.css', () => ({}))

describe('App', () => {
  it('renders home page', () => {
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>,
    )
    expect(
      screen.getByRole('heading', { name: /welcome to culture ui/i }),
    ).toBeInTheDocument()

  })
})
