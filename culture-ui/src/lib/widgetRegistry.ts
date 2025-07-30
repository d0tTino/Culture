export interface WidgetMeta {
  scriptUrl?: string
}

export interface WidgetInfo extends WidgetMeta {
  name: string
}

export interface WidgetRegistry {
  register<P = unknown>(name: string, component: React.ComponentType<P>, meta?: WidgetMeta): void
  get(name: string): React.ComponentType<unknown> | undefined
  list(): string[]
}

class Registry implements WidgetRegistry {
  private widgets = new Map<string, React.ComponentType<unknown>>()

  register<P = unknown>(name: string, component: React.ComponentType<P>, meta?: WidgetMeta) {
    this.widgets.set(name, component as React.ComponentType<unknown>)
    try {
      void fetch('/api/widgets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, ...(meta ?? {}) }),
      }).catch(() => {})
    } catch {
      // ignore registration errors
    }
  }

  get(name: string) {
    return this.widgets.get(name)
  }

  list() {
    return Array.from(this.widgets.keys())
  }
}

export const widgetRegistry = new Registry()
export const registerWidget = widgetRegistry.register.bind(widgetRegistry)
export const getWidget = widgetRegistry.get.bind(widgetRegistry)
export const listWidgets = widgetRegistry.list.bind(widgetRegistry)

export async function loadRemoteWidgets() {
  try {
    const res = await fetch('/api/widgets')
    const data = (await res.json()) as { widgets: WidgetInfo[] }
    await Promise.all(
      data.widgets.map(async (w) => {
        if (w.scriptUrl) {
          await import(/* @vite-ignore */ w.scriptUrl)
        }
      }),
    )
  } catch {
    // ignore failures
  }
}


