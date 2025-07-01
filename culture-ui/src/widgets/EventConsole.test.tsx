import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import EventConsole from './EventConsole'
import { MockEventSource, resetMockSources } from '../lib/testUtils'

afterEach(() => {
  resetMockSources()
})

describe('EventConsole', () => {
  it('renders incoming events', () => {
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource

    render(<EventConsole />)
    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage('{"foo": 1}')
    })

    expect(screen.getByTestId('events').textContent).toContain('"foo": 1')
  })

  it('filters events using search input', async () => {
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource

    render(<EventConsole />)
    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage('{"type":"a"}')
    })
    act(() => {
      es.emitMessage('{"type":"b"}')
    })

    await userEvent.type(screen.getByLabelText('search'), 'b')

    const text = screen.getByTestId('events').textContent || ''
    expect(text).toMatch(/"type"\s*:\s*"b"/)
    expect(text).not.toMatch(/"type"\s*:\s*"a"/)
  })
})
