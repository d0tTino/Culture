import { render, screen } from '@testing-library/react'
import AgentDataOverview from './pages/AgentDataOverview'

vi.mock('./App.css', () => ({}))
vi.mock('flexlayout-react/style/light.css', () => ({}))

describe('AgentDataOverview widget', () => {
  it('renders Agent Data Overview component', () => {
    render(<AgentDataOverview />)
    expect(
      screen.getByRole('heading', { name: /agent data overview/i }),
    ).toBeInTheDocument()
  })
})
