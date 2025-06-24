/* eslint-disable @typescript-eslint/no-explicit-any */
import { act, render, screen } from '@testing-library/react'
import KpiCard from './KpiCard'
import { MockEventSource, MockWebSocket, resetMockSources } from '../lib/testUtils'

afterEach(() => {
  resetMockSources()
})

describe('KpiCard', () => {
  type GlobalWithSources = typeof globalThis & {
    EventSource?: unknown
    WebSocket?: unknown
  }

  it('renders KPI card', () => {
  ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
    MockEventSource

    render(<KpiCard />)
    expect(screen.getByTestId('kpi-card')).toBeInTheDocument()
    expect(screen.getByTestId('kpi-value').textContent).toBe('0')
  })

  it('updates chart on SSE messages', () => {
  ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
    MockEventSource

    render(<KpiCard />)
    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage('{"data":{"value":1}}')
    })
    act(() => {
      es.emitMessage('{"data":{"value":2}}')
    })
    expect(screen.getByTestId('kpi-value').textContent).toBe('2')
    expect(screen.getByTestId('chart').querySelectorAll('span')).toHaveLength(2)
  })

  it('updates chart using WebSocket fallback', () => {
  ;(globalThis as unknown as { EventSource?: typeof EventSource }).EventSource =
    undefined
  ;(globalThis as unknown as { WebSocket?: typeof WebSocket }).WebSocket =
    MockWebSocket

    render(<KpiCard />)
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.sendMessage('{"data":{"value":5}}')
    })
    expect(screen.getByTestId('kpi-value').textContent).toBe('5')
    expect(screen.getByTestId('chart').querySelectorAll('span')).toHaveLength(1)
  })
})

