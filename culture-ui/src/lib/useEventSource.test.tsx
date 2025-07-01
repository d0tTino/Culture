import { act, render, screen, waitFor } from '@testing-library/react'
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest'
import { spawn, type ChildProcess } from 'child_process'
import EventSourcePolyfill from 'eventsource'
import path from 'node:path'
import { createServer, type AddressInfo } from 'node:net'
import { useEventSource } from './useEventSource'
import {
  MockEventSource,
  MockWebSocket,
  resetMockSources,
} from './testUtils'

async function getPort(): Promise<number> {
  return await new Promise((resolve, reject) => {
    const srv = createServer()
    srv.listen(0, () => {
      const { port } = srv.address() as AddressInfo
      srv.close((err) => (err ? reject(err) : resolve(port)))
    })
    srv.on('error', reject)
  })
}


function TestComponent() {
  const event = useEventSource()
  return <div data-testid="value">{event ? JSON.stringify(event) : ''}</div>
}

afterEach(() => {
  resetMockSources()
})

describe('useEventSource', () => {
  it('connects via EventSource and receives messages', () => {
    ;(
      globalThis as unknown as {
        EventSource: typeof EventSource
      }
    ).EventSource = MockEventSource as unknown as typeof EventSource

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
        EventSource: typeof EventSource | undefined
        WebSocket: typeof WebSocket
      }
    ).EventSource = undefined
    ;(
      globalThis as unknown as {
        EventSource: typeof EventSource | undefined
        WebSocket: typeof WebSocket
      }
    ).WebSocket = MockWebSocket as unknown as typeof WebSocket

    render(<TestComponent />)
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.sendMessage('{"foo":"bar"}')
    })
    expect(screen.getByTestId('value').textContent).toBe(JSON.stringify({ foo: 'bar' }))
  })

  it('falls back to WebSocket when EventSource constructor throws', () => {
    ;(
      globalThis as unknown as {
        EventSource: typeof EventSource
        WebSocket: typeof WebSocket
      }
    ).EventSource = class {
      constructor() {
        throw new Error('fail')
      }
    } as unknown as typeof EventSource
    ;(
      globalThis as unknown as {
        EventSource: typeof EventSource
        WebSocket: typeof WebSocket
      }
    ).WebSocket = MockWebSocket as unknown as typeof WebSocket

    render(<TestComponent />)
    const ws = MockWebSocket.instances[0]
    act(() => {
      ws.sendMessage('{"b":2}')
    })
    expect(screen.getByTestId('value').textContent).toBe(JSON.stringify({ b: 2 }))
  })

  it('falls back to WebSocket on EventSource error', () => {
    ;(
      globalThis as unknown as {
        EventSource: typeof EventSource
        WebSocket: typeof WebSocket
      }
    ).EventSource = MockEventSource as unknown as typeof EventSource
    ;(
      globalThis as unknown as {
        EventSource: typeof EventSource
        WebSocket: typeof WebSocket
      }
    ).WebSocket = MockWebSocket as unknown as typeof WebSocket

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
        EventSource: typeof EventSource
      }
    ).EventSource = MockEventSource as unknown as typeof EventSource

    const { unmount } = render(<TestComponent />)
    const es = MockEventSource.instances[0]
    unmount()
    expect(es.closed).toBe(true)
  })
})

describe('useEventSource integration with FastAPI', () => {
  let port: number
  let server: ChildProcess | undefined
  let serverAvailable = true

  beforeAll(async () => {
    try {
      port = await getPort()
      server = spawn('python', ['scripts/simple_event_app.py', String(port)], {
        cwd: path.resolve(__dirname, '../../..'),
        stdio: 'ignore',
        env: { ...process.env, PYTHONPATH: path.resolve(__dirname, '../../..') },
      })
      // wait for server to be ready
      for (let i = 0; i < 50; i++) {
        try {
          const res = await fetch(`http://127.0.0.1:${port}/health`)
          if (res.ok) return
        } catch {
          /* ignore */
        }
        await new Promise((r) => setTimeout(r, 100))
      }
      serverAvailable = false
    } catch {
      serverAvailable = false
    }
  })

  afterAll(() => {
    if (server) server.kill()
  })

  it('receives events from the backend', async () => {
    if (!serverAvailable) {
      console.warn('Skipping integration test: server not available')
      return
    }
    ;(
      globalThis as unknown as {
        EventSource: typeof EventSourcePolyfill
      }
    ).EventSource = class extends EventSourcePolyfill {
      constructor(url: string, opts?: EventSourceInit) {
        super(`http://127.0.0.1:${port}${url}`, opts)
      }
    }

    render(<TestComponent />)
    await waitFor(() => {
      expect(screen.getByTestId('value').textContent).toBe(
        JSON.stringify({ event_type: 'test', data: { value: 1 } }),
      )
    })
  })
})
