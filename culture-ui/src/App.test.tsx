import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import App from './App'

vi.mock('./App.css', () => ({}))
vi.mock('flexlayout-react/style/light.css', () => ({}))
vi.mock('./pages/MissionOverview', () => ({ default: () => <div /> }))

describe('App', () => {
  it('renders home page', () => {
    render(<App />)
    expect(
      screen.getByRole('heading', { name: /welcome to culture ui/i }),
    ).toBeInTheDocument()

  })
})
