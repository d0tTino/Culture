export class MockEventSource {
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

export class MockWebSocket {
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

export function resetMockSources() {
  MockEventSource.instances = []
  MockWebSocket.instances = []
  type GlobalWithSources = typeof globalThis & {
    EventSource?: unknown
    WebSocket?: unknown
  }
  ;(globalThis as GlobalWithSources).EventSource = undefined
  ;(globalThis as GlobalWithSources).WebSocket = undefined
}

