import { render, screen } from '@testing-library/react'
import MemoryTimeline from './MemoryTimeline'

describe('MemoryTimeline', () => {
  it('shows memory events', () => {
    render(
      <MemoryTimeline events={[{ type: 'memory_prune', step: 1 }]} />,
    )
    expect(screen.getByText('memory_prune (step 1)')).toBeInTheDocument()
  })
})
