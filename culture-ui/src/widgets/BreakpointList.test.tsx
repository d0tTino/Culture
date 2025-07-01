import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import BreakpointList from './BreakpointList'
import { MockEventSource, resetMockSources } from '../lib/testUtils'

afterEach(() => {
  resetMockSources()
  vi.restoreAllMocks()
})

describe('BreakpointList', () => {
  it('POSTs selected tags to /control', async () => {
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource
    const fetchMock = vi.fn(() => Promise.resolve({ ok: true } as Response))
    global.fetch = fetchMock as unknown as typeof fetch

    render(<BreakpointList />)
    await userEvent.click(screen.getByLabelText('violence'))
    await userEvent.click(screen.getByText('Save'))

    expect(fetchMock).toHaveBeenCalledWith(
      '/control',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: 'set_breakpoints', tags: ['violence'] }),
      }),
    )
  })

  it('shows toast on breakpoint hit event', () => {
    ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
      MockEventSource as unknown as typeof EventSource

    render(<BreakpointList />)
    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage(
        '{"event_type":"breakpoint_hit","data":{"tags":["nsfw"],"step":1}}',
      )
    })

    expect(screen.getByTestId('toast').textContent).toContain('nsfw')
  })
})
