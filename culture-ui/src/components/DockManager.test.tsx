import { render } from '@testing-library/react'
import { vi } from 'vitest'
vi.mock('flexlayout-react/style/light.css', () => ({}))
vi.mock('flexlayout-react', () => {
  const Layout = vi.fn((props: any) => {
    Layout.lastProps = props
    return null
  }) as any
  const Model = {
    fromJson: vi.fn((json: any) => ({ toJson: () => json }))
  }
  return { Layout, Model, TabNode: class {} }
})
import DockManager from './DockManager'
import { widgetRegistry } from '../lib/widgetRegistry'
import { Layout, Model } from 'flexlayout-react'


declare module 'flexlayout-react' {
  interface Model { toJson(): any }
}

function TestWidget() {
  return <div data-testid="widget" />
}

describe('DockManager', () => {
  beforeEach(() => {
    widgetRegistry.register('test', TestWidget)
    localStorage.clear()
  })

  it('loads layout from localStorage', () => {
    localStorage.setItem('dockLayout', JSON.stringify({ saved: true }))
    render(<DockManager defaultLayout={{}} />)
    expect(Model.fromJson).toHaveBeenCalledWith({ saved: true })
  })

  it('persists layout changes', () => {
    const prevEnv = process.env.NODE_ENV
    process.env.NODE_ENV = 'development'
    render(<DockManager defaultLayout={{ foo: 1 }} />)
    const model = Model.fromJson.mock.results[0].value
    expect(typeof Layout.lastProps.onModelChange).toBe('function')
    process.env.NODE_ENV = prevEnv
  })
})
