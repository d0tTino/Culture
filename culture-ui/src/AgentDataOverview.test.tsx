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

describe('AgentDataOverview', () => {
  it('renders KPI metrics and chart', () => {
    render(<AgentDataOverview />)
    expect(screen.getByRole('heading', { name: /agent data overview/i })).toBeInTheDocument()
    expect(screen.getByTestId('active-agents')).toHaveTextContent('9')
    expect(screen.getByTestId('total-messages')).toHaveTextContent('100')
    expect(screen.getByTestId('avg-agents')).toHaveTextContent('7')
    expect(screen.getByTestId('agents-line-chart')).toBeInTheDocument()
  })
})
