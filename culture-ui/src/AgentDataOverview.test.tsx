import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import App from './App'

vi.mock('./App.css', () => ({}))

describe('AgentDataOverview routing', () => {
  it('loads Agent Data Overview page when navigating', () => {
    window.history.pushState({}, '', '/agent-data')
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>,
    )
    expect(screen.getByRole('heading', { name: /agent data overview/i })).toBeInTheDocument()
  })
})
