import { describe, it, expect } from 'vitest'
import { widgetRegistry } from './widgetRegistry'
import type { FC } from 'react'

const Dummy: FC = () => null

describe('widgetRegistry', () => {
  it('registers and retrieves components', () => {
    const name = `widget_${Math.random()}`
    widgetRegistry.register(name, Dummy)
    const retrieved = widgetRegistry.get(name)
    expect(retrieved).toBe(Dummy)
    expect(widgetRegistry.list()).toContain(name)
  })
})
