import '@testing-library/jest-dom'

class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

;(global as typeof global & { ResizeObserver?: typeof ResizeObserver }).ResizeObserver = ResizeObserver

import { vi } from 'vitest'
vi.mock('flexlayout-react/style/light.css', () => ({}), { virtual: true })
