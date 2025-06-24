import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import Home from './pages/Home'

vi.mock('./App.css', () => ({}))
vi.mock('flexlayout-react/style/light.css', () => ({}))

describe('App', () => {
  it('renders home page', () => {
    render(<Home />)
    expect(
      screen.getByRole('heading', { name: /welcome to culture ui/i }),
    ).toBeInTheDocument()
  })
})
