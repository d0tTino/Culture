import '@testing-library/jest-dom'

class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

;(global as typeof global & { ResizeObserver?: typeof ResizeObserver }).ResizeObserver = ResizeObserver

// Provide a minimal EventSource implementation for tests
class MockEventSource {
  constructor(url: string) {
    void url
  }
  onmessage: ((ev: MessageEvent) => void) | null = null
  onerror: ((ev: Event) => void) | null = null
  addEventListener() {}
  removeEventListener() {}
  close() {}
}

;(globalThis as typeof globalThis & { EventSource?: typeof EventSource }).EventSource =
  MockEventSource as unknown as typeof EventSource

import { vi } from 'vitest'
vi.mock('flexlayout-react/style/light.css', () => ({}))
