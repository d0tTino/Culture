import { render } from '@testing-library/react'
import { vi } from 'vitest'
import NetworkWeb from './widgets/NetworkWeb'

vi.mock(
  'react-force-graph-2d',
  () => ({
    __esModule: true,
    default: () => <canvas />,
  }),
  { virtual: true },
)

describe('NetworkWeb', () => {
  it('renders without crashing', () => {
    const { container } = render(<NetworkWeb />)
    expect(container.querySelector('canvas')).toBeInTheDocument()
  })
})
