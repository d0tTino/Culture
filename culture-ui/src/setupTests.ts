import '@testing-library/jest-dom'
import ResizeObserverPolyfill from 'resize-observer-polyfill'

// Vitest doesn't provide ResizeObserver by default. Some UI components
// (like Recharts' ResponsiveContainer) rely on it for measurements.
// Assign a polyfill so these components work in the test environment.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
;(globalThis as any).ResizeObserver = ResizeObserverPolyfill
