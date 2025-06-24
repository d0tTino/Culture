import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'
import TimelineWidget from './TimelineWidget'

class MockEventSource {
  close() {}
}

vi.mock('../lib/useEventSource', () => ({
  useEventSource: vi.fn(() => null),
}))

describe('TimelineWidget', () => {
  it('renders slider control', () => {
    ;(globalThis as unknown as { EventSource: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource

    render(<TimelineWidget />)
    expect(screen.getByRole('slider')).toBeInTheDocument()
  })
})
