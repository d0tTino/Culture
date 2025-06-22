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
    ;(globalThis as any).EventSource = MockEventSource
    render(<TimelineWidget />)
    expect(screen.getByRole('slider')).toBeInTheDocument()
  })
})
