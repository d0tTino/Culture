import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { useEventSource } from './useEventSource'

class MockEventSource {
  static instances: MockEventSource[] = []
  url: string
  onmessage: ((ev: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  closed = false

  constructor(url: string) {
    this.url = url
    MockEventSource.instances.push(this)
  }

  emitMessage(data: string) {
    this.onmessage?.({ data } as MessageEvent)
  }

  emitError() {
    this.onerror?.()
  }

  close() {
    this.closed = true
  }
}

class MockWebSocket {
  static instances: MockWebSocket[] = []
  url: string
  onmessage: ((ev: MessageEvent) => void) | null = null
  closed = false

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }

  sendMessage(data: string) {
    this.onmessage?.({ data } as MessageEvent)
  }

  close() {
    this.closed = true
  }
}

function TestComponent() {
  const event = useEventSource()
  return <div data-testid="value">{event ? JSON.stringify(event) : ''}</div>
}

afterEach(() => {
  MockEventSource.instances = []
  MockWebSocket.instances = []
  ;(
    globalThis as unknown as {
      EventSource: unknown
      WebSocket: unknown
    }
  ).EventSource = undefined
  ;(
    globalThis as unknown as {
      EventSource: unknown
      WebSocket: unknown
    }
  ).WebSocket = undefined
})

describe('useEventSource', () => {
  it('connects via EventSource and receives messages', () => {
    ;(
      globalThis as unknown as {
        EventSource: typeof MockEventSource
      }
    ).EventSource = MockEventSource

    render(<TestComponent />)
    const es = MockEventSource.instances[0]
    act(() => {
      es.emitMessage('{"hello":1}')
    })
    expect(screen.getByTestId('value').textContent).toBe(JSON.stringify({ hello: 1 }))
    act(() => {})
  })

  it('falls back to WebSocket when EventSource is unavailable', () => {
    ;(
      globalThis as unknown as {
        EventSource: undefined
        WebSocket: typeof MockWebSocket
      }
    ).EventSource = undefined
    ;(
      globalThis as unknown as {
        EventSource: undefined
        WebSocket: typeof MockWebSocket
      }
    ).WebSocket = MockWebSocket

    render(<TestComponent />)
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.sendMessage('{"foo":"bar"}')
    })
    expect(screen.getByTestId('value').textContent).toBe(JSON.stringify({ foo: 'bar' }))
  })

  it('falls back to WebSocket on EventSource error', () => {
    ;(
      globalThis as unknown as {
        EventSource: typeof MockEventSource
        WebSocket: typeof MockWebSocket
      }
    ).EventSource = MockEventSource
    ;(
      globalThis as unknown as {
        EventSource: typeof MockEventSource
        WebSocket: typeof MockWebSocket
      }
    ).WebSocket = MockWebSocket

    render(<TestComponent />)
    const es = MockEventSource.instances[0]
    act(() => {
      es.emitError()
    })
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.sendMessage('{"x":1}')
    })
    expect(es.closed).toBe(true)
    expect(screen.getByTestId('value').textContent).toBe(JSON.stringify({ x: 1 }))
  })

  it('cleans up connections on unmount', () => {
    ;(
      globalThis as unknown as {
        EventSource: typeof MockEventSource
      }
    ).EventSource = MockEventSource

    const { unmount } = render(<TestComponent />)
    const es = MockEventSource.instances[0]
    unmount()
    expect(es.closed).toBe(true)
  })
})
