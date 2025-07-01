export class MockEventSource {
  static instances: MockEventSource[] = []
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSED = 2
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

export class MockWebSocket {
  static instances: MockWebSocket[] = []
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
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

interface GlobalWithSources {
  EventSource?: typeof EventSource
  WebSocket?: typeof WebSocket
}

export function resetMockSources() {
  MockEventSource.instances = []
  MockWebSocket.instances = []
  ;(globalThis as unknown as GlobalWithSources).EventSource = undefined
  ;(globalThis as unknown as GlobalWithSources).WebSocket = undefined

}

