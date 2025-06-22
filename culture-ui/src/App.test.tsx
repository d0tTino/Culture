import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import App from './App'

vi.mock('./App.css', () => ({}))

describe('App', () => {
  it('renders agent overview heading', () => {
    render(<App />)
    expect(
      screen.getByRole('heading', { name: /agent data overview/i }),
    ).toBeInTheDocument()
  })
})
