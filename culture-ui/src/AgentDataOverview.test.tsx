import { render, screen } from '@testing-library/react'
import App from './App'

vi.mock('./App.css', () => ({}))
vi.mock('flexlayout-react/style/light.css', () => ({}))
vi.mock('./pages/MissionOverview', () => ({ default: () => <div /> }))

describe('AgentDataOverview widget', () => {
  it('renders Agent Data Overview component', () => {
    render(<App />)
    expect(
      screen.getByRole('heading', { name: /agent data overview/i }),
    ).toBeInTheDocument()
  })
})
