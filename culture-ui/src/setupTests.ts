import '@testing-library/jest-dom'

class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

;(global as typeof global & { ResizeObserver?: typeof ResizeObserver }).ResizeObserver = ResizeObserver
