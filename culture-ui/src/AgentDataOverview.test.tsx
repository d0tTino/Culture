import { render, screen } from '@testing-library/react'
import App from './App'

vi.mock('./App.css', () => ({}))
vi.mock('flexlayout-react/style/light.css', () => ({}))

describe('AgentDataOverview widget', () => {
  it('renders Agent Data Overview component', () => {
    render(<App />)
    expect(
      screen.getByRole('heading', { name: /agent data overview/i }),
    ).toBeInTheDocument()
  })
})
