import { render, screen } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import DockManager from './DockManager'
import { widgetRegistry } from '../lib/widgetRegistry'
import type { FC, ReactNode } from 'react'
import type { IJsonModel } from 'flexlayout-react'

vi.mock('flexlayout-react', () => ({
  Layout: ({ children }: { children?: ReactNode }) => <div>{children}</div>,
  Model: { fromJson: vi.fn(() => ({ toJson: vi.fn(() => ({})) })) },
  TabNode: class {},
}))

import { Model } from 'flexlayout-react'

const fromJson = Model.fromJson as ReturnType<typeof vi.fn>

const defaultLayout = { global: {}, layout: {} } as IJsonModel

const registry = widgetRegistry as unknown as { widgets: Map<string, FC> }

describe('DockManager', () => {
  beforeEach(() => {
    localStorage.clear()
    fromJson.mockClear()
    registry.widgets.clear()
  })

  it('restores layout from localStorage and renders registered widgets', () => {
    const name = 'TestWidget'
    const Widget: FC = () => <div data-testid="test-widget">test</div>
    widgetRegistry.register(name, Widget)
    const saved = { layout: { type: 'row', children: [] } }
    localStorage.setItem('dockLayout', JSON.stringify(saved))
    render(<DockManager defaultLayout={defaultLayout} />)
    expect(fromJson).toHaveBeenCalledWith(saved)
    expect(screen.getByTestId('test-widget')).toBeInTheDocument()
  })
})
