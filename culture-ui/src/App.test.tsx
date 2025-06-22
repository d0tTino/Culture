import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import App from './App'

vi.mock('./App.css', () => ({}))

describe('App', () => {
  it('renders mission overview heading by default', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: /mission overview/i })).toBeInTheDocument()
  })
})
