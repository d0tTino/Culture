import { render, screen } from '@testing-library/react'
import AgentDataOverview from './pages/AgentDataOverview'

vi.mock('./App.css', () => ({}))
vi.mock('flexlayout-react/style/light.css', () => ({}))
vi.mock('recharts', () => ({
  ResponsiveContainer: (props: any) => <div>{props.children}</div>,
  LineChart: (props: any) => <svg>{props.children}</svg>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
}))

describe('AgentDataOverview widget', () => {
  it('renders Agent Data Overview component', () => {
    render(<AgentDataOverview />)
    expect(
      screen.getByRole('heading', { name: /agent data overview/i }),
    ).toBeInTheDocument()

  })
})
